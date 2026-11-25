"""Observador del corpus: convierte texto en observaciones estructuradas.

Es el equivalente de `_stateLog` + `_observeStateSequence` + `_mineFromStateLog`
de SymbolicAbstractionLayer.js, pero la entrada es texto del corpus, no
estados bioquímicos.

Mantiene:
  - observations: ventanas de texto procesadas
  - vocabulary: frecuencia de tokens significativos
  - transitions: contadores de pares (A→B) con delay en caracteres
  - sequences: ventanas deslizantes de tokens
  - state_log: registro plano para mining tipo SAL
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import tiktoken


# Stopwords ES + EN (recortadas del SAL original)
STOPWORDS = frozenset("""
the and for with from this that these those some any all each other such
very just only also even still then than too much many more most less not
yes can will would could should have has had been being was were are is am
be do does did done doing
los las una uno unos unas el la lo un con del por que sin sobre bajo entre
hacia hasta desde según durante mediante para como tras ante contra mas pero
sino aunque mientras porque pues ni y o u e menos muy aqui ahi alli alla aca
asi tan tanto cuanto cuando donde siempre nunca jamas todavia aun ya luego
despues antes ahora entonces ademas tambien tampoco solo solamente casi
este esta esto ese esa eso aquel aquella estos estas esos esas cada otro otra
otros otras mismo misma mismos mismas algun alguna algunos algunas ningun
ninguna todo toda todos todas nada algo alguien nadie quien quienes cual
cuales cuyo cuya yo tu el ella nosotros vosotros ellos ellas me te se nos os
mi mis tus su sus nuestro nuestra vuestro vuestra ser estar tener hacer ir
venir poder deber querer saber decir ver dar poner parecer soy eres es somos
sois son era eran estoy estas esta estamos estais estan tengo tienes tiene
tenemos teneis tienen hago haces hace hacemos haceis hacen voy vas va vamos
vais van sea sean fuera fueron haber habia
""".split())


MIN_TOKEN_LEN = 4
MIN_OCCURRENCES_FOR_SYMBOL = 3
MIN_OCCURRENCES_FOR_RULE = 3
MIN_TRANSITION_DELAY_CHARS = 10
MAX_TRANSITION_DELAY_CHARS = 500
MAX_OBSERVATIONS = 5000
MAX_STATE_LOG = 1000
MAX_SEQUENCE_LEN = 50
MAX_TRANSITIONS = 50000
MAX_DELAYS_PER_TRANSITION = 100
# Cota por token para evitar O(n²) en `_min_delay`
MAX_POSITIONS_PER_TOKEN = 50


@dataclass
class Observation:
    text: str
    tokens: list[str]
    at: float
    ref: str = ""
    chunk_id: int = 0


@dataclass
class Transition:
    a: str
    b: str
    count: int = 0
    delays: list[int] = field(default_factory=list)

    def record(self, delay: int) -> None:
        self.count += 1
        self.delays.append(delay)
        if len(self.delays) > MAX_DELAYS_PER_TRANSITION:
            self.delays.pop(0)

    @property
    def mean_delay(self) -> float:
        return sum(self.delays) / len(self.delays) if self.delays else 0.0


# Token de "palabra": letras Unicode de longitud >= 3 (sin dígitos ni _).
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


class CorpusObserver:
    """Observa el corpus y extrae símbolos, secuencias y transiciones.

    Uso:
        obs = CorpusObserver()
        for chunk in chunks:
            obs.observe(chunk)
    """

    def __init__(self, enc_name: str = "gpt2"):
        self.enc_name = enc_name
        self._enc = None
        self.observations: list[Observation] = []
        self.vocabulary: Counter = Counter()
        self.transitions: dict[tuple[str, str], Transition] = {}
        self.sequences: list[list[str]] = []
        self.state_log: list[dict] = []
        self.chunk_counter = 0
        self.metrics = {
            "observations": 0,
            "tokens_total": 0,
            "tokens_unique": 0,
            "transitions": 0,
            "sequences": 0,
        }

    def _encoding(self):
        if self._enc is None:
            self._enc = tiktoken.get_encoding(self.enc_name)
        return self._enc

    # ---------------------------------------------------------- tokenización
    def tokenize(self, text: str) -> list[str]:
        """Tokeniza a palabras significativas (sin stopwords, min length)."""
        if not text:
            return []
        raw = _WORD_RE.findall(text.lower())
        return [t for t in raw if t not in STOPWORDS and len(t) >= MIN_TOKEN_LEN]

    # ---------------------------------------------------------- observación
    def observe(self, text: str, ref: str = "") -> Observation:
        """Registra una ventana de texto y actualiza contadores."""
        self.chunk_counter += 1
        tokens = self.tokenize(text)
        obs = Observation(
            text=text,
            tokens=tokens,
            at=time.time(),
            ref=ref,
            chunk_id=self.chunk_counter,
        )
        self.observations.append(obs)
        if len(self.observations) > MAX_OBSERVATIONS:
            self.observations.pop(0)

        for t in tokens:
            self.vocabulary[t] += 1

        # Secuencias de hasta 10 tokens (ventana deslizante)
        for i in range(len(tokens)):
            window = tokens[i : i + 10]
            if len(window) >= 2:
                self.sequences.append(window)
        if len(self.sequences) > MAX_SEQUENCE_LEN * 100:
            self.sequences = self.sequences[-MAX_SEQUENCE_LEN * 100 :]

        # Transiciones: pares (A, B) con delay en caracteres
        self._record_transitions(text, tokens)

        # state_log: registro plano para mining tipo SAL
        self.state_log.append({
            "at": obs.at,
            "chunk_id": self.chunk_counter,
            "facts": list(tokens),
        })
        if len(self.state_log) > MAX_STATE_LOG:
            self.state_log.pop(0)

        self.metrics["observations"] += 1
        self.metrics["tokens_total"] += len(tokens)
        self.metrics["tokens_unique"] = len(self.vocabulary)
        return obs

    def _record_transitions(self, text: str, tokens: list[str]) -> None:
        """Cuenta pares (A→B) con delay entre ocurrencias en el texto."""
        positions: dict[str, list[int]] = defaultdict(list)
        for m in _WORD_RE.finditer(text.lower()):
            tok = m.group()
            if tok in STOPWORDS or len(tok) < MIN_TOKEN_LEN:
                continue
            lst = positions[tok]
            # Cota por token para mantener `_min_delay` acotado.
            if len(lst) < MAX_POSITIONS_PER_TOKEN:
                lst.append(m.start())

        seen_pairs: set[tuple[str, str]] = set()
        for i, a in enumerate(tokens):
            for b in tokens[i + 1 : i + 20]:
                if a == b:
                    continue
                pair = (a, b)
                if pair in seen_pairs:
                    continue
                delay = self._min_delay(positions[a], positions[b])
                if delay is None:
                    continue
                if (
                    delay < MIN_TRANSITION_DELAY_CHARS
                    or delay > MAX_TRANSITION_DELAY_CHARS
                ):
                    continue
                seen_pairs.add(pair)
                if pair not in self.transitions:
                    if len(self.transitions) >= MAX_TRANSITIONS:
                        continue
                    self.transitions[pair] = Transition(a=a, b=b)
                self.transitions[pair].record(delay)

        self.metrics["transitions"] = len(self.transitions)

    @staticmethod
    def _min_delay(a_pos: list[int], b_pos: list[int]) -> int | None:
        if not a_pos or not b_pos:
            return None
        best = None
        for pa in a_pos:
            for pb in b_pos:
                d = pb - pa
                if d > 0 and (best is None or d < best):
                    best = d
        return best

    # ---------------------------------------------------------- símbolos
    def candidate_symbols(self) -> list[tuple[str, int]]:
        """Tokens con suficientes ocurrencias para ser símbolo."""
        return [
            (t, c)
            for t, c in self.vocabulary.items()
            if c >= MIN_OCCURRENCES_FOR_SYMBOL
        ]

    def candidate_transitions(self) -> list[tuple[str, str, int]]:
        """Pares (A, B, count) con suficientes ocurrencias para ser regla."""
        out = []
        for (a, b), tr in self.transitions.items():
            if tr.count >= MIN_OCCURRENCES_FOR_RULE:
                out.append((a, b, tr.count))
        return out

    def get_state(self) -> dict:
        return {
            "observations": len(self.observations),
            "vocabulary_size": len(self.vocabulary),
            "transitions": len(self.transitions),
            "sequences": len(self.sequences),
            "state_log": len(self.state_log),
            "metrics": dict(self.metrics),
            "top_tokens": self.vocabulary.most_common(10),
        }

    def reset(self) -> None:
        self.observations = []
        self.vocabulary = Counter()
        self.transitions = {}
        self.sequences = []
        self.state_log = []
        self.chunk_counter = 0
        self.metrics = {
            "observations": 0,
            "tokens_total": 0,
            "tokens_unique": 0,
            "transitions": 0,
            "sequences": 0,
        }
