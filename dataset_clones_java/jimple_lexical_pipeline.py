#!/usr/bin/env python3
"""
jimple_lexical_pipeline.py
Pipeline: Análisis léxico de patrones Jimple-like sobre código Java
→ XGBoost Classifier

DIFERENCIA vs jimple_xgboost_pipeline.py:
  - NO requiere javac ni Soot
  - Extrae patrones que Jimple generaría directamente del texto Java
  - 100% cobertura del dataset (vs ~15-25% del pipeline Soot)
  - Comparable directamente con TF-IDF y AST en el paper

JUSTIFICACIÓN CIENTÍFICA:
  Los patrones léxicos extraídos son proxies directos de las instrucciones
  Jimple: 'new ' → 'new' Jimple, 'instanceof' → 'instanceof' Jimple,
  llamadas a métodos → 'virtualinvoke/staticinvoke', etc.
  Esta representación captura la misma señal estructural sin requerir
  compilación, superando la limitación de dependencias externas del dataset.

Uso:
  python jimple_lexical_pipeline.py              # dataset completo
  python jimple_lexical_pipeline.py --test 50    # prueba con 50 pares
"""

import argparse
import json
import re
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN — misma estructura que los otros pipelines
# ──────────────────────────────────────────────────────────────
SEED        = 42
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "model_ready_balanced"
RESULTS_DIR = SCRIPT_DIR / "results" / "jimple_lexical"

# ──────────────────────────────────────────────────────────────
# FEATURES — 32 patrones léxicos que corresponden a instrucciones Jimple
#
# Cada feature tiene su equivalente Jimple documentado:
# ──────────────────────────────────────────────────────────────
#
# GRUPO 1: Tipos de invocación (→ invoke types en Jimple)
# virtualinvoke   ← obj.method()      llamadas sobre instancias
# staticinvoke    ← Class.method()    llamadas estáticas
# interfaceinvoke ← interface.method() llamadas a interfaces
# specialinvoke   ← super.method()    constructores y super
# dynamicinvoke   ← lambda / stream   expresiones funcionales
#
# GRUPO 2: Control de flujo (→ if/goto/switch en Jimple)
# if_stmt         ← if(...)
# for_loop        ← for(...)
# while_loop      ← while(...)
# ternary         ← cond ? a : b
# switch_stmt     ← switch(...)
# return_stmt     ← return ...
# break_continue  ← break / continue
#
# GRUPO 3: Manejo de excepciones (→ throw/catch en Jimple)
# throw_stmt      ← throw new ...
# try_catch       ← try { ... } catch
# finally_block   ← finally { ... }
#
# GRUPO 4: Creación de objetos y tipos (→ new/newarray/checkcast)
# new_object      ← new ClassName()
# new_array       ← new Type[]
# instanceof_check← instanceof
# cast_expr       ← (Type) expr
#
# GRUPO 5: Acceso a datos (→ field access en Jimple)
# field_access    ← this.field / obj.field
# array_access    ← arr[i]
# null_check      ← == null / != null
# string_concat   ← str + "..." o String.format
#
# GRUPO 6: Operaciones aritméticas y lógicas (→ binop en Jimple)
# arithmetic      ← + - * / %
# bitwise         ← & | ^ ~ << >>
# comparison      ← == != < > <= >=
# logical         ← && ||
#
# GRUPO 7: Tipos de retorno y modificadores (señales estructurales)
# returns_void    ← void method
# returns_bool    ← boolean method
# returns_object  ← Object/class return
# is_static       ← static method
# is_override     ← @Override
# has_generics    ← List<T>, Map<K,V>
# sync_block      ← synchronized
# ──────────────────────────────────────────────────────────────

FEATURE_DEFS = [
    # (nombre_feature, patron_regex, jimple_equivalente)
    # GRUPO 1: Invocaciones
    ("invoke_instance",  r'\w+\.\w+\s*\(',          "virtualinvoke"),
    ("invoke_static",    r'\b[A-Z]\w*\.\w+\s*\(',   "staticinvoke"),
    ("invoke_super",     r'\bsuper\s*\.',             "specialinvoke"),
    ("invoke_this",      r'\bthis\s*\.',              "specialinvoke"),
    ("lambda_expr",      r'->',                       "dynamicinvoke"),
    ("method_ref",       r'::',                       "dynamicinvoke"),
    ("stream_api",       r'\.(stream|filter|map|collect|reduce|forEach)\s*\(', "dynamicinvoke"),

    # GRUPO 2: Control de flujo
    ("if_stmt",          r'\bif\s*\(',                "if Jimple"),
    ("for_loop",         r'\bfor\s*\(',               "goto + if Jimple"),
    ("while_loop",       r'\bwhile\s*\(',             "goto + if Jimple"),
    ("ternary",          r'\?[^:]+:',                 "if Jimple"),
    ("switch_stmt",      r'\bswitch\s*\(',            "switch Jimple"),
    ("return_stmt",      r'\breturn\b',               "return Jimple"),
    ("break_continue",   r'\b(break|continue)\b',    "goto Jimple"),

    # GRUPO 3: Excepciones
    ("throw_stmt",       r'\bthrow\b',                "throw Jimple"),
    ("try_catch",        r'\b(try|catch)\s*[\(\{]',  "catch Jimple"),
    ("finally_block",    r'\bfinally\s*\{',           "catch Jimple"),

    # GRUPO 4: Creación y tipos
    ("new_object",       r'\bnew\s+[A-Z]\w*\s*\(',   "new Jimple"),
    ("new_array",        r'\bnew\s+\w+\s*\[',         "newarray Jimple"),
    ("instanceof_check", r'\binstanceof\b',            "instanceof Jimple"),
    ("cast_expr",        r'\([A-Z][A-Za-z<>]*\)\s*\w',"checkcast Jimple"),

    # GRUPO 5: Acceso a datos
    ("field_access",     r'\b(this|self)\.\w+',       "field ref Jimple"),
    ("array_access",     r'\w+\s*\[\s*\w',            "arrayref Jimple"),
    ("null_check",       r'[!=]=\s*null',              "= null Jimple"),
    ("string_concat",    r'"\s*\+|String\.format',    "staticinvoke Jimple"),

    # GRUPO 6: Operaciones
    ("arithmetic",       r'[\+\-\*/%]=?\s',           "add/sub/mul Jimple"),
    ("bitwise",          r'[&|^~]|<<|>>',             "and/or/xor Jimple"),
    ("comparison",       r'[=!<>]=|[<>](?!=)',        "cmp Jimple"),
    ("logical",          r'&&|\|\|',                  "if + goto Jimple"),

    # GRUPO 7: Modificadores estructurales
    ("is_static",        r'\bstatic\b',               "staticinvoke signal"),
    ("is_override",      r'@Override',                "virtualinvoke signal"),
    ("has_generics",     r'<[A-Z]\w*>',               "checkcast signal"),
    ("sync_block",       r'\bsynchronized\b',         "monitorenter Jimple"),
]

FEATURE_NAMES = [f[0] for f in FEATURE_DEFS]
PATTERNS      = [(f[0], re.compile(f[1])) for f in FEATURE_DEFS]
N_FEATURES    = len(FEATURE_DEFS)  # 32

# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS — idéntica al baseline del equipo
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
            except json.JSONDecodeError:
                continue
            if required_fields - set(obj):
                continue
            records.append({
                "func1": obj["func1"],
                "func2": obj["func2"],
                "clone_type": obj["clone_type"],
            })
    return records

# ──────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES LÉXICOS JIMPLE-LIKE
# ──────────────────────────────────────────────────────────────
def extract_jimple_lexical_features(code: str) -> np.ndarray:
    """
    Extrae el vector de 32 features léxicos de un método Java.
    Cada feature es el conteo normalizado de ocurrencias del patrón
    correspondiente — proxy directo de instrucciones Jimple.
    """
    if not code:
        return np.zeros(N_FEATURES, dtype=np.float32)

    lines      = code.splitlines()
    total_lines = max(len(lines), 1)
    vec        = np.zeros(N_FEATURES, dtype=np.float32)

    for i, (name, pattern) in enumerate(PATTERNS):
        count    = sum(1 for l in lines if pattern.search(l))
        vec[i]   = count / total_lines   # normalizado por longitud

    return vec

def extract_pair_features(records: list, split_name: str) -> tuple:
    """
    Para cada par (func1, func2) genera el vector simétrico:
      X_pair = [|v1-v2|, v1⊙v2, cosine_sim(v1,v2)]
    Shape: (n_pairs, N_FEATURES*2 + 1) = (n_pairs, 65)

    Mismo diseño simétrico que TF-IDF y AST del equipo.
    Cobertura: 100% — no requiere compilación.
    """
    n = len(records)
    X = np.zeros((n, N_FEATURES * 2 + 1), dtype=np.float32)
    y = [rec["clone_type"] for rec in records]

    for i, rec in enumerate(records):
        v1 = extract_jimple_lexical_features(rec["func1"])
        v2 = extract_jimple_lexical_features(rec["func2"])

        norm1      = np.linalg.norm(v1) + 1e-9
        norm2      = np.linalg.norm(v2) + 1e-9
        cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))

        X[i] = np.concatenate([
            np.abs(v1 - v2),   # N_FEATURES abs diff
            v1 * v2,            # N_FEATURES hadamard
            [cosine_sim]        # 1 cosine
        ])

    print(f"  ✅ [{split_name}] {n}/{n} pares procesados (100.0% cobertura)")
    return X, y

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Jimple Lexical + XGBoost — 100% dataset coverage"
    )
    parser.add_argument("--test", type=int, default=None,
                        help="Modo prueba: N pares por split")
    args   = parser.parse_args()
    test_n = args.test

    print("=" * 65)
    print("  PIPELINE: Jimple Léxico + XGBoost"
          + (f"  [PRUEBA — {test_n} pares]" if test_n else ""))
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Features    : {N_FEATURES} patrones léxicos Jimple-like")
    print(f"  Vector/par  : {N_FEATURES*2+1} features (abs_diff + product + cosine)")
    print(f"  Cobertura   : 100% — sin dependencia de javac/Soot")
    print(f"  data_dir    : {DATA_DIR}")
    print("=" * 65)

    for fname in ["train_balanced.jsonl", "valid_balanced.jsonl",
                  "test_balanced.jsonl"]:
        if not (DATA_DIR / fname).exists():
            print(f"❌ No se encontró {DATA_DIR / fname}")
            return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──
    print("\n📂 Cargando datos...")
    train_data = load_jsonl(str(DATA_DIR / "train_balanced.jsonl"))
    valid_data = load_jsonl(str(DATA_DIR / "valid_balanced.jsonl"))
    test_data  = load_jsonl(str(DATA_DIR / "test_balanced.jsonl"))
    if test_n:
        train_data = train_data[:test_n]
        valid_data = valid_data[:test_n]
        test_data  = test_data[:test_n]
    print(f"   Train:{len(train_data)} Valid:{len(valid_data)} "
          f"Test:{len(test_data)}")

    # ── Extraer features ──
    print("\n🔧 Extrayendo features léxicos Jimple-like...")
    t0 = datetime.now()
    X_train, y_train = extract_pair_features(train_data, "Train")
    X_valid, y_valid = extract_pair_features(valid_data, "Valid")
    X_test,  y_test  = extract_pair_features(test_data,  "Test")
    elapsed = (datetime.now() - t0).seconds
    print(f"\n   Shapes → Train:{X_train.shape} "
          f"Valid:{X_valid.shape} Test:{X_test.shape}")
    print(f"   Tiempo de extracción: {elapsed}s")

    # ── Encoding ──
    le = LabelEncoder()
    le.fit(y_train)
    y_tr = le.transform(y_train)
    y_v  = le.transform(y_valid)
    y_te = le.transform(y_test)

    # ── XGBoost — configuración idéntica al equipo ──
    print("\n🌳 Entrenando XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
        tree_method="hist",
        n_jobs=-1
    )
    clf.fit(X_train, y_tr,
            eval_set=[(X_valid, y_v)],
            verbose=50)
    best_round = clf.best_iteration

    # ── Evaluación ──
    print("\n📊 Evaluando en Test...")
    y_pred      = le.inverse_transform(clf.predict(X_test))
    accuracy    = accuracy_score(y_test, y_pred) * 100
    macro_f1    = f1_score(y_test, y_pred, average="macro",
                           zero_division=0) * 100
    weighted_f1 = f1_score(y_test, y_pred, average="weighted",
                           zero_division=0) * 100
    report      = classification_report(y_test, y_pred,
                                        digits=4, zero_division=0)

    print("\n" + "=" * 65)
    print("  RESULTADOS" + (" [PRUEBA]" if test_n else ""))
    print("=" * 65)
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  Macro F1    : {macro_f1:.2f}%")
    print(f"  Weighted F1 : {weighted_f1:.2f}%")
    print(f"  Mejor iter. : {best_round}")
    print(f"  Cobertura   : 100% ({len(test_data)}/{len(test_data)} pares)")
    print(f"\n{report}")

    # ── Guardar ──
    suffix   = f"_test{test_n}" if test_n else ""
    out_path = RESULTS_DIR / f"jimple_lexical_results{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pipeline": "jimple_lexical",
            "mode": f"test_{test_n}" if test_n else "full",
            "features_count": N_FEATURES * 2 + 1,
            "feature_names": FEATURE_NAMES,
            "coverage": "100%",
            "best_xgb_iteration": best_round,
            "metrics": {
                "accuracy":    round(accuracy, 4),
                "macro_f1":    round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
            },
            "classification_report": report,
            "splits": {
                "train": len(train_data),
                "valid": len(valid_data),
                "test":  len(test_data),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados: {out_path}")

    # ── Tabla comparativa completa ──
    if not test_n:
        print("\n📋 TABLA COMPARATIVA COMPLETA:")
        print(f"{'Modelo':<30} {'Accuracy':>10} {'Macro F1':>10} "
              f"{'Features':>10} {'Cobertura':>10}")
        print("-" * 74)
        print(f"{'TF-IDF + XGBoost':<30} {'84.54%':>10} {'87.30%':>10} "
              f"{'2,001':>10} {'100%':>10}")
        print(f"{'AST + XGBoost':<30} {'77.74%':>10} {'76.10%':>10} "
              f"{'58':>10} {'99%':>10}")
        print(f"{'TF-IDF+AST Híbrido':<30} {'92.44%':>10} {'93.82%':>10} "
              f"{'2,059':>10} {'99%':>10}")
        print(f"{'Jimple Léxico + XGBoost':<30} {accuracy:>9.2f}% "
              f"{macro_f1:>9.2f}% {N_FEATURES*2+1:>10} {'100%':>10}")
        print(f"{'Jimple Soot + XGBoost':<30} {'⏳':>10} {'⏳':>10} "
              f"{'37':>10} {'~20%':>10}")
    print("=" * 65)

if __name__ == "__main__":
    main()
