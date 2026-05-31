#!/usr/bin/env python3
"""
Diagnóstico específico de Soot.
Toma los primeros 3 métodos que SÍ compilaron con javac
y ejecuta Soot mostrando el output completo.
Corre en dataset_clones_java/
"""
import json
import subprocess
import tempfile
from pathlib import Path

SOOT_JAR = Path("soot.jar")
DATA     = Path("model_ready_balanced/train_balanced.jsonl")

def wrap_for_soot(method_code: str, class_name: str) -> str:
    body = (method_code or "").strip()
    return (
        "import java.io.*;\nimport java.util.*;\nimport java.util.stream.*;\n"
        "import java.nio.*;\nimport java.nio.channels.*;\nimport java.nio.file.*;\n"
        "import java.net.*;\nimport java.util.concurrent.*;\n"
        "@SuppressWarnings(\"all\")\n"
        f"public class {class_name} {{\n"
        "    private String logFile = \"\";\n"
        "    private String rotateDest = null;\n"
        "    private void printFile(String s, Object o) {}\n"
        f"{body}\n"
        "}\n"
    )

records = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        if len(records) >= 5:
            break
        try:
            records.append(json.loads(line.strip()))
        except:
            continue

with tempfile.TemporaryDirectory() as tmpdir:
    work     = Path(tmpdir)
    jimple_d = work / "jimple"
    jimple_d.mkdir()

    # Compilar individualmente y quedarnos con los que sí pasan
    compiled = []
    for i, rec in enumerate(records):
        for suffix, code in [("f1", rec["func1"]), ("f2", rec["func2"])]:
            cname     = f"C{i}_{suffix}"
            java_code = wrap_for_soot(code, cname)
            java_path = work / f"{cname}.java"
            java_path.write_text(java_code, encoding="utf-8", errors="replace")

            r = subprocess.run(
                ["javac", "-nowarn", "--release", "11", "-d", str(work), str(java_path)],
                capture_output=True, text=True, timeout=30
            )
            if (work / f"{cname}.class").exists():
                compiled.append(cname)
                print(f"✅ javac OK: {cname}")
            else:
                print(f"❌ javac FAIL: {cname}")

    if not compiled:
        print("\n❌ Ningún archivo compiló — revisa el wrapper")
    else:
        print(f"\n📦 {len(compiled)} clases compiladas: {compiled}")
        print("\n" + "="*60)
        print("EJECUTANDO SOOT — output completo:")
        print("="*60)

        # Probar con UNA sola clase primero
        test_class = compiled[0]
        cmd = [
            "java", "-jar", str(SOOT_JAR),
            "-cp", str(work),
            "-f", "jimple",
            "-d", str(jimple_d),
            "-pp",
            "-allow-phantom-refs",
            "-v",          # verbose — muestra todo
            test_class
        ]
        print(f"Comando: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print("--- STDOUT ---")
        print(result.stdout[:2000] if result.stdout else "(vacío)")
        print("\n--- STDERR ---")
        print(result.stderr[:2000] if result.stderr else "(vacío)")
        print(f"\nReturn code: {result.returncode}")

        jimple_file = jimple_d / f"{test_class}.jimple"
        if jimple_file.exists():
            print(f"\n✅ .jimple generado! ({jimple_file.stat().st_size} bytes)")
            print("\nContenido:")
            print(jimple_file.read_text()[:500])
        else:
            print(f"\n❌ .jimple NO generado en {jimple_d}")
            print(f"   Archivos en jimple_d: {list(jimple_d.iterdir())}")
            print(f"   Archivos en work: {[f.name for f in work.iterdir()]}")
