import os
import json
import random
from collections import defaultdict

def main():
    random.seed(42)  # Seed para reproducibilidad
    
    base_dir = "/Users/danielfloresrojas/Documents/Dataset_Java/dataset_clones_java/model_ready"
    files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    
    # 2. GLOBAL COUNTS
    counts_by_type = defaultdict(int)
    total_pairs = 0
    
    # 3. RANDOM SAMPLING
    # Vamos a guardar todos los ejemplos en memoria o hacer reservoir sampling.
    # Dado que jsonl puede ser grande, vamos a leer y guardar los índices, o mejor,
    # simplemente guardaremos todos los ejemplos temporalmente si cabe en memoria,
    # pero para eficiencia usaremos reservoir sampling por cada tipo.
    
    samples_by_type = defaultdict(list)
    k = 3 # 3 muestras por tipo
    
    print("Reading files and computing counts...")
    for filename in files:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ctype = obj.get("clone_type", "UNKNOWN")
                    
                    # Update counts
                    counts_by_type[ctype] += 1
                    total_pairs += 1
                    
                    # Reservoir sampling
                    if len(samples_by_type[ctype]) < k:
                        samples_by_type[ctype].append(obj)
                    else:
                        j = random.randint(0, counts_by_type[ctype] - 1)
                        if j < k:
                            samples_by_type[ctype][j] = obj
                except Exception as e:
                    pass

    # Preparar tabla
    summary_md = "| Etiqueta (clone_type) | Cantidad de Pares |\n"
    summary_md += "|-----------------------|-------------------|\n"
    for ctype, count in sorted(counts_by_type.items()):
        summary_md += f"| {ctype} | {count:,} |\n"
    summary_md += f"| **TOTAL GLOBAL** | **{total_pairs:,}** |\n"
    
    # 4. REPORT GENERATION
    report_path = "/Users/danielfloresrojas/Documents/Dataset_Java/QA_Validation_Report.md"
    print("Generating report at:", report_path)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Validación QA del Dataset de Clones\n\n")
        f.write("## 1. Conteos Globales (Estadísticas)\n\n")
        f.write(summary_md)
        f.write("\n\n")
        
        f.write("## 2. Muestreo Aleatorio (Extracción Manual)\n\n")
        
        # Orden preferido para mostrar
        order = ["T0", "T1", "T2", "VST3", "ST3", "MT3", "T4"]
        # Añadir cualquier otro que pueda existir
        for c in sorted(counts_by_type.keys()):
            if c not in order:
                order.append(c)
                
        for ctype in order:
            if ctype not in samples_by_type or len(samples_by_type[ctype]) == 0:
                continue
            
            f.write(f"### Muestras de Tipo {ctype}\n\n")
            samples = samples_by_type[ctype]
            for i, sample in enumerate(samples, 1):
                f.write(f"#### Muestra {i} (id1: `{sample['id1']}` | id2: `{sample['id2']}`)\n\n")
                f.write("**Fragmento 1 (`func1`):**\n")
                f.write("```java\n")
                f.write(sample["func1"].strip() + "\n")
                f.write("```\n\n")
                f.write("**Fragmento 2 (`func2`):**\n")
                f.write("```java\n")
                f.write(sample["func2"].strip() + "\n")
                f.write("```\n\n")
                f.write("---\n\n")

    print("SUMMARY_TABLE_START")
    print(summary_md)
    print("SUMMARY_TABLE_END")

if __name__ == "__main__":
    main()
