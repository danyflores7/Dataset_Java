#!/usr/bin/env python3
"""
jimple_xgboost_pipeline.py
Pipeline: Java → Jimple IR (Soot 4.3.0) → XGBoost Classifier
Detección multiclase de clones de código Java

Uso:
  python jimple_xgboost_pipeline.py              # dataset completo
  python jimple_xgboost_pipeline.py --test 50    # prueba con 50 pares por split
  python jimple_xgboost_pipeline.py --test 200   # prueba con 200 pares por split
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────
SEED       = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "model_ready_balanced"
SOOT_JAR   = SCRIPT_DIR / "soot.jar"
RESULTS_DIR= SCRIPT_DIR / "results" / "jimple_xgboost"

JIMPLE_FEATURES = [
    "virtualinvoke", "interfaceinvoke", "staticinvoke", "specialinvoke",
    "dynamicinvoke", "checkcast", "instanceof", "newarray", "new ",
    "throw", "catch", "goto", "if ", "return", "= null",
    "lengthof", "= (int)", "= (double)",
]
N_FEATURES = len(JIMPLE_FEATURES)  # 18

# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    required_fields = {"func1", "func2", "clone_type"}
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}:{line_number}: JSON inválido omitido ({exc})")
                continue
            missing = required_fields - set(obj)
            if missing:
                continue
            records.append({
                "func1": obj["func1"],
                "func2": obj["func2"],
                "clone_type": obj["clone_type"],
            })
    return records

# ──────────────────────────────────────────────────────────────
# PREPARACIÓN JAVA PARA SOOT
# El dataset contiene métodos COMPLETOS con firma (public void X()...)
# Solo hay que envolverlos en una clase — NO dentro de otro método
# ──────────────────────────────────────────────────────────────
def wrap_for_soot(method_code: str, class_name: str) -> str:
    """
    Envuelve el método en una clase Java válida para Soot.
    Agrega imports comunes para maximizar la tasa de compilación.
    """
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

# ──────────────────────────────────────────────────────────────
# COMPILACIÓN BATCH: escribe TODOS los .java y compila de una vez
# Esto es la optimización clave — en lugar de llamar javac
# 35,000 veces, lo llamamos una vez por split con todos los archivos
# ──────────────────────────────────────────────────────────────
def compile_batch(records: list, work_dir: Path) -> dict:
    """
    Escribe todos los .java de un split y los compila en una sola
    llamada a javac. Mucho más rápido que compilar uno por uno.
    Retorna dict {class_name: compiled_ok}
    """
    java_files = []
    class_names = {}

    for i, rec in enumerate(records):
        for suffix, code in [("f1", rec["func1"]), ("f2", rec["func2"])]:
            cname     = f"C{i}_{suffix}"
            java_code = wrap_for_soot(code, cname)
            java_path = work_dir / f"{cname}.java"
            java_path.write_text(java_code, encoding="utf-8", errors="replace")
            java_files.append(str(java_path))
            class_names[cname] = False  # default: no compilado

    if not java_files:
        return class_names

    print(f"    Compilando {len(java_files)} archivos Java con javac...")
    # javac acepta múltiples archivos — una sola llamada
    result = subprocess.run(
        ["javac", "-d", str(work_dir)] + java_files,
        capture_output=True, text=True, timeout=300
    )
    # Marcar los que sí compilaron (existe el .class)
    for cname in class_names:
        if (work_dir / f"{cname}.class").exists():
            class_names[cname] = True

    ok = sum(class_names.values())
    print(f"    ✅ javac: {ok}/{len(java_files)} archivos compilados "
          f"({ok/len(java_files)*100:.1f}%)")
    return class_names

# ──────────────────────────────────────────────────────────────
# SOOT BATCH: procesa todos los .class de una vez
# ──────────────────────────────────────────────────────────────
def run_soot_batch(class_names: list, bytecode_dir: Path,
                   jimple_out_dir: Path) -> dict:
    """
    Llama a Soot una sola vez con todas las clases compiladas.
    Mucho más eficiente que una llamada por clase.
    """
    if not class_names:
        return {}

    print(f"    Ejecutando Soot sobre {len(class_names)} clases...")
    cmd = (
        ["java", "-jar", str(SOOT_JAR),
         "-cp", str(bytecode_dir),
         "-f", "jimple",
         "-d", str(jimple_out_dir),
         "-pp", "-allow-phantom-refs"]
        + class_names
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Verificar qué .jimple se generaron
    success = {}
    for cname in class_names:
        success[cname] = (jimple_out_dir / f"{cname}.jimple").exists()

    ok = sum(success.values())
    print(f"    ✅ Soot: {ok}/{len(class_names)} archivos .jimple generados "
          f"({ok/len(class_names)*100:.1f}%)")
    return success

# ──────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES DESDE .jimple
# ──────────────────────────────────────────────────────────────
def extract_features_from_jimple(jimple_path: Path) -> np.ndarray:
    try:
        text  = jimple_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        total = max(len(lines), 1)
        return np.array(
            [sum(1 for l in lines if feat in l) / total
             for feat in JIMPLE_FEATURES],
            dtype=np.float32
        )
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)

# ──────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL POR SPLIT — versión batch
# ──────────────────────────────────────────────────────────────
def extract_jimple_pair_features(records: list, split_name: str) -> tuple:
    """
    Procesa todos los pares de un split en modo batch:
    1. Escribe todos los .java y compila con UNA llamada a javac
    2. Procesa todos los .class con UNA llamada a Soot
    3. Extrae features de todos los .jimple
    Devuelve (X, y, tasa_exito)
    """
    n = len(records)
    X = np.zeros((n, N_FEATURES * 2 + 1), dtype=np.float32)
    y = [rec["clone_type"] for rec in records]

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir   = Path(tmpdir)
        jimple_out = work_dir / "jimple"
        jimple_out.mkdir()

        # Paso 1: compilar todo en batch
        compiled = compile_batch(records, work_dir)

        # Paso 2: Soot en batch sobre los que compilaron
        ok_classes = [c for c, ok in compiled.items() if ok]
        soot_ok    = run_soot_batch(ok_classes, work_dir, jimple_out)

        # Paso 3: extraer features por par
        success_count = 0
        for i in range(n):
            cn1 = f"C{i}_f1"
            cn2 = f"C{i}_f2"

            j1 = jimple_out / f"{cn1}.jimple"
            j2 = jimple_out / f"{cn2}.jimple"

            v1 = extract_features_from_jimple(j1) if j1.exists() else np.zeros(N_FEATURES, dtype=np.float32)
            v2 = extract_features_from_jimple(j2) if j2.exists() else np.zeros(N_FEATURES, dtype=np.float32)

            if j1.exists() and j2.exists():
                success_count += 1

            abs_diff   = np.abs(v1 - v2)
            product    = v1 * v2
            norm1      = np.linalg.norm(v1) + 1e-9
            norm2      = np.linalg.norm(v2) + 1e-9
            cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))
            X[i]       = np.concatenate([abs_diff, product, [cosine_sim]])

    rate = success_count / n * 100
    print(f"  ✅ [{split_name}] Pares con Jimple completo: "
          f"{success_count}/{n} ({rate:.1f}%)")
    return X, y, rate

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jimple + XGBoost pipeline")
    parser.add_argument("--test", type=int, default=None,
                        help="Modo prueba: usa solo los primeros N pares por split")
    args = parser.parse_args()

    test_mode = args.test is not None
    test_n    = args.test if test_mode else 0

    print("=" * 65)
    print("  PIPELINE: Jimple IR + XGBoost"
          + (f"  [MODO PRUEBA — {test_n} pares]" if test_mode else ""))
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directorio : {SCRIPT_DIR}")
    print(f"  soot.jar   : {'✅ encontrado' if SOOT_JAR.exists() else '❌ NO encontrado'}")
    print(f"  data_dir   : {DATA_DIR}")
    print("=" * 65)

    if not SOOT_JAR.exists():
        print("\n❌ ERROR: soot.jar requerido.")
        print("   curl -L -o soot.jar https://repo1.maven.org/maven2/org/soot-oss/"
              "soot/4.3.0/soot-4.3.0-jar-with-dependencies.jar")
        return

    for fname in ["train_balanced.jsonl", "valid_balanced.jsonl", "test_balanced.jsonl"]:
        if not (DATA_DIR / fname).exists():
            print(f"\n❌ ERROR: No se encontró {DATA_DIR / fname}")
            return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──
    print("\n📂 Cargando datos...")
    train_data = load_jsonl(str(DATA_DIR / "train_balanced.jsonl"))
    valid_data = load_jsonl(str(DATA_DIR / "valid_balanced.jsonl"))
    test_data  = load_jsonl(str(DATA_DIR / "test_balanced.jsonl"))

    # Aplicar límite si es modo prueba
    if test_mode:
        train_data = train_data[:test_n]
        valid_data = valid_data[:test_n]
        test_data  = test_data[:test_n]
        print(f"   ⚡ MODO PRUEBA: {test_n} pares por split")

    print(f"   Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")

    # ── Extraer features Jimple en batch ──
    print("\n🔧 Extrayendo features Jimple (modo batch — mucho más rápido)...")
    t0 = datetime.now()

    X_train, y_train, rate_train = extract_jimple_pair_features(train_data, "Train")
    X_valid, y_valid, rate_valid = extract_jimple_pair_features(valid_data, "Valid")
    X_test,  y_test,  rate_test  = extract_jimple_pair_features(test_data,  "Test")

    elapsed = (datetime.now() - t0).seconds
    print(f"\n   Tiempo de extracción: {elapsed}s")
    print(f"   Shape Train : {X_train.shape}")
    print(f"   Shape Valid : {X_valid.shape}")
    print(f"   Shape Test  : {X_test.shape}")

    # ── Encoding ──
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)
    y_test_enc  = le.transform(y_test)

    # ── XGBoost ──
    print("\n🌳 Entrenando XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="mlogloss",
        early_stopping_rounds=15, tree_method="hist", n_jobs=-1
    )
    clf.fit(X_train, y_train_enc,
            eval_set=[(X_valid, y_valid_enc)], verbose=50)

    # ── Evaluación ──
    print("\n📊 Evaluando en Test...")
    y_pred     = le.inverse_transform(clf.predict(X_test))
    accuracy   = accuracy_score(y_test, y_pred) * 100
    macro_f1   = f1_score(y_test, y_pred, average="macro") * 100
    weighted_f1= f1_score(y_test, y_pred, average="weighted") * 100
    report     = classification_report(y_test, y_pred, digits=4)

    print("\n" + "=" * 65)
    print("  RESULTADOS" + (" [MODO PRUEBA]" if test_mode else ""))
    print("=" * 65)
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  Macro F1    : {macro_f1:.2f}%")
    print(f"  Weighted F1 : {weighted_f1:.2f}%")
    print(f"  Soot éxito  : Train {rate_train:.1f}% | "
          f"Valid {rate_valid:.1f}% | Test {rate_test:.1f}%")
    print(f"\n{report}")

    # ── Guardar ──
    suffix = f"_test{test_n}" if test_mode else ""
    out = {
        "timestamp": datetime.now().isoformat(),
        "mode": f"test_{test_n}" if test_mode else "full",
        "features": N_FEATURES * 2 + 1,
        "soot_success_rates": {"train": rate_train, "valid": rate_valid, "test": rate_test},
        "metrics": {"accuracy": round(accuracy,4), "macro_f1": round(macro_f1,4),
                    "weighted_f1": round(weighted_f1,4)},
        "classification_report": report,
    }
    out_path = RESULTS_DIR / f"jimple_results{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Resultados guardados: {out_path}")

    # ── Tabla comparativa ──
    if not test_mode:
        print("\n📋 TABLA COMPARATIVA:")
        print(f"{'Modelo':<28} {'Accuracy':>10} {'Macro F1':>10} {'Features':>10}")
        print("-" * 62)
        print(f"{'TF-IDF + XGBoost':<28} {'84.54%':>10} {'87.30%':>10} {'2,001':>10}")
        print(f"{'AST + XGBoost':<28} {'77.74%':>10} {'76.10%':>10} {'58':>10}")
        print(f"{'TF-IDF+AST Híbrido':<28} {'92.44%':>10} {'93.82%':>10} {'2,059':>10}")
        print(f"{'Jimple + XGBoost':<28} {accuracy:>9.2f}% {macro_f1:>9.2f}% "
              f"{N_FEATURES*2+1:>10}")
    print("=" * 65)

if __name__ == "__main__":
    main()
