#!/usr/bin/env python3
"""
Inspecciona los primeros 3 registros del dataset para ver
exactamente qué contiene func1 y func2
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "model_ready_balanced/train_balanced.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        rec = json.loads(line.strip())
        print(f"\n{'='*60}")
        print(f"REGISTRO {i+1} — clone_type: {rec['clone_type']}")
        print(f"{'='*60}")
        print(f"--- func1 ({len(rec['func1'])} chars) ---")
        print(rec['func1'][:500])
        print(f"\n--- func2 ({len(rec['func2'])} chars) ---")
        print(rec['func2'][:500])

