"""RTModelWrapper: SNNLM + ResonantVocab. Encapsula el flujo completo."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .resonant_vocab import ResonantVocab


class RTModelWrapper(nn.Module):
    """SNNLM con embeddings cuantizados por el RT.

    Sustituye el `tok_emb` del SNNLM por `rv.embed_and_quantize`.
    Delega atributos al SNNLM (config, neural_memory, abstraction_head, etc.).
    """

    def __init__(
        self,
        snnlm: nn.Module,
        rv: ResonantVocab,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        if snnlm.config.vocab_size != rv.vocab_size:
            raise ValueError(
                f"SNNLM vocab_size={snnlm.config.vocab_size} != "
                f"RV vocab_size={rv.vocab_size}. "
                f"Set GPTConfig.vocab_size = rv.vocab_size when using RT."
            )
        self.snnlm = snnlm
        self.rv = rv
        self.aux_loss_weight = aux_loss_weight

    @property
    def config(self):
        return self.snnlm.config

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------ forward
    def forward(
        self,
        ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches=None,
        start_pos: int = 0,
        update_memory: Optional[bool] = None,
        update_vocab: Optional[bool] = None,
    ):
        # ── FIX v4.2: actualizar vocab en TRAIN, no en EVAL ──
        update_vocab_eff = (
            update_vocab if update_vocab is not None else self.training
        )
        quantized_emb, _, aux_loss = self.rv.embed_and_quantize(
            ids, update=update_vocab_eff
        )

        out = self.snnlm(
            idx=ids,
            targets=targets,
            kv_caches=kv_caches,
            start_pos=start_pos,
            update_memory=update_memory,
            precomputed_embeddings=quantized_emb,
        )

        if targets is not None:
            if kv_caches is not None:
                logits, loss, new_kv = out
                loss = loss + self.aux_loss_weight * aux_loss
                return logits, loss, new_kv
            logits, loss = out
            loss = loss + self.aux_loss_weight * aux_loss
            return logits, loss
        return out

    # ------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        update_memory: bool = False,  # ── FIX: default seguro ──
    ) -> torch.Tensor:
        self.eval()
        if max_new_tokens <= 0:
            return idx

        seq_len = self.config.seq_len
        if idx.size(1) > seq_len:
            idx = idx[:, -seq_len:]

        # ── FIX v4.2: LIF temporal no es compatible con KV-cache ──
        if (
            use_cache
            and self.config.use_spiking
            and self.config.snn_mode == "temporal"
        ):
            use_cache = False

        if use_cache and idx.size(1) < seq_len:
            kv_caches = [None] * len(self.snnlm.blocks)
            logits, _, _ = self(
                idx, kv_caches=kv_caches, start_pos=0,
                update_memory=update_memory,
            )
            pos = idx.size(1)
        else:
            use_cache = False
            kv_caches = None
            logits, _ = self(idx, update_memory=update_memory)
            pos = idx.size(1)

        next_logits = logits[:, -1, :]

        for step in range(max_new_tokens):
            next_id = self.snnlm._sample_next(
                next_logits, temperature, top_k, top_p
            )
            idx = torch.cat([idx, next_id], dim=1)
            if step == max_new_tokens - 1:
                break

            if use_cache and pos + 1 <= seq_len:
                next_emb, _, _ = self.rv.embed_and_quantize(next_id, update=False)
                out = self.snnlm(
                    idx=next_id,
                    kv_caches=kv_caches,
                    start_pos=pos,
                    update_memory=update_memory,
                    precomputed_embeddings=next_emb,
                )
                logits, _, kv_caches = out
                pos += 1
                next_logits = logits[:, -1, :]
            else:
                use_cache = False
                idx_cond = idx[:, -seq_len:]
                logits, _ = self(idx_cond, update_memory=update_memory)
                next_logits = logits[:, -1, :]

        return idx

    # ------------------------------------------------------------ utils
    def encode(self, text: str) -> list[int]:
        return self.rv.encode(text)

    def decode(self, ids) -> str:
        return self.rv.decode(ids)

    def memory_state_for_checkpoint(self) -> dict:
        state = self.snnlm.memory_state_for_checkpoint() if hasattr(
            self.snnlm, "memory_state_for_checkpoint"
        ) else {}
        state["resonant_vocab"] = self.rv.state_for_checkpoint()
        return state

    def load_memory_state_from_checkpoint(self, data: dict, device):
        if not data:
            return
        # ── FIX v4.2: no mutar el dict del caller ──
        rv_state = data.get("resonant_vocab")
        if rv_state is not None:
            self.rv.load_state_from_checkpoint(rv_state, device)
        if hasattr(self.snnlm, "load_memory_state_from_checkpoint"):
            self.snnlm.load_memory_state_from_checkpoint(data, device)

    def __getattr__(self, name: str):
        # ── FIX v4.2: guarda contra recursión durante __init__/deepcopy ──
        modules = self.__dict__.get("_modules")
        if not modules or "snnlm" not in modules:
            raise AttributeError(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(modules["snnlm"], name)
