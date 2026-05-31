#!/usr/bin/env python3
"""
Diagnóstico: intenta compilar los primeros 3 métodos del dataset
y muestra exactamente qué error da javac.
Corre esto en dataset_clones_java/
"""
import json
import subprocess
import tempfile
from pathlib import Path

DATA = Path("model_ready_balanced/train_balanced.jsonl")

def wrap_for_soot(method_code: str, class_name: str) -> str:
    body = (method_code or "").strip()
    return (
        "import java.io.*;\n"
        "import java.util.*;\n"
        "import java.util.stream.*;\n"
        "import java.nio.*;\n"
        "import java.nio.channels.*;\n"
        "import java.nio.file.*;\n"
        f"public class {class_name} {{\n"
        f"{body}\n"
        f"}}\n"
    )

records = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        if len(records) >= 3:
            break
        try:
            records.append(json.loads(line.strip()))
        except:
            continue

with tempfile.TemporaryDirectory() as tmpdir:
    work = Path(tmpdir)

    for i, rec in enumerate(records):
        for suffix, code in [("f1", rec["func1"]), ("f2", rec["func2"])]:
            cname     = f"C{i}_{suffix}"
            java_code = wrap_for_soot(code, cname)
            java_path = work / f"{cname}.java"
            java_path.write_text(java_code, encoding="utf-8", errors="replace")

            print(f"\n{'='*60}")
            print(f"Compilando {cname} (clone_type={rec['clone_type']})")
            print(f"{'='*60}")
            print("--- Java generado ---")
            print(java_code[:400])

            result = subprocess.run(
                ["javac", "-d", str(work), str(java_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print("✅ COMPILÓ OK")
            else:
                print("❌ ERROR javac:")
                print(result.stderr[:600])
