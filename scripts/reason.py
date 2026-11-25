"""Consulta el grafo de razonamiento sin generar texto.

Ejemplos:
    snnlm-reason --graph checkpoints/graph.pt --stats
    snnlm-reason --graph checkpoints/graph.pt --apply "quick" "fox" "lazy"
    snnlm-reason --corpus data/ --mine --save checkpoints/graph.pt
"""
from __future__ import annotations

import argparse
import json
import os

from snn_lm_v3 import (
    load_texts,
    CorpusObserver,
    ReasoningGraph,
    save_graph,
)
from snn_lm_v3.reasoning import load_graph


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", default=None, help="Cargar grafo desde disco")
    src.add_argument("--corpus", default=None, help="Construir grafo desde corpus")
    p.add_argument("--save", default=None, help="Guardar grafo tras minar")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--stats", action="store_true")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--apply", nargs="+", default=None,
                   help="Facts a consultar en el grafo")
    p.add_argument("--json", action="store_true", help="Salida JSON")
    return p.parse_args()


def main():
    args = parse_args()

    if args.graph:
        graph = load_graph(args.graph)
        print(f"Grafo cargado: {graph.get_state()['rules_count']} reglas")
    else:
        observer = CorpusObserver(enc_name="gpt2")
        docs = load_texts([args.corpus])
        for i, doc in enumerate(docs):
            for j in range(0, len(doc), args.chunk_size):
                observer.observe(doc[j : j + args.chunk_size], ref=f"ref{i}_{j}")
        graph = ReasoningGraph(cooldown_steps=3)
        n = graph.mine_from_corpus(observer)
        print(f"Reglas minadas: {n}")
        if args.save:
            save_graph(graph, args.save)
            print(f"Grafo guardado en {args.save}")

    state = graph.get_state()

    if args.stats:
        if args.json:
            print(json.dumps(state, indent=2, default=str))
        else:
            print("\n== Estado del grafo ==")
            for k, v in state.items():
                if k in ("top_rules", "metrics"):
                    continue
                print(f"  {k}: {v}")
            print("  metrics:")
            for k, v in state["metrics"].items():
                print(f"    {k}: {v}")
            print(f"  top_{args.top}_rules:")
            for r in graph.get_state()["top_rules"][: args.top]:
                print(
                    f"    {r['antecedent']} → {r['consequent']} "
                    f"(conf={r['confidence']}, uses={r['uses']})"
                )

    if args.apply:
        consequences = graph.apply(args.apply)
        if args.json:
            print(json.dumps(consequences, indent=2, default=str))
        else:
            print(f"\n== Consecuencias para {args.apply} ==")
            if not consequences:
                print("  (ninguna)")
            for c in consequences:
                print(
                    f"  {c['antecedent']} → {c['consequent']} "
                    f"(conf={c['confidence']})"
                )


if __name__ == "__main__":
    main()
