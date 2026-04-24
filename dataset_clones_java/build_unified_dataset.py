import os
import json
import random
from collections import defaultdict

def get_clone_type(syntactic_type, sim_token):
    if syntactic_type == "1":
        return "T1"
    elif syntactic_type == "2":
        return "T2"
    elif syntactic_type == "3":
        try:
            sim = float(sim_token)
        except ValueError:
            return "UNKNOWN"
        
        if sim >= 0.90:
            return "VST3"
        elif sim >= 0.70:
            return "ST3"
        elif sim >= 0.50:
            return "MT3"
        else:
            return "WT3" # Weakly T3 / T4
    return "UNKNOWN"

def split_java_functions(content):
    # Split by at least 2 consecutive newlines, trying to find two distinct blocks of code
    # Many files have "\n\n" or "\n\n\n" separating the two methods.
    parts = [p.strip() for p in content.split('\n\n') if p.strip()]
    if len(parts) >= 2:
        # The first part is func1, the rest is func2 (or vice versa)
        # We will just split in half roughly if there are multiple parts, or rely on the largest split.
        # Typically there are just 2 methods.
        mid = len(parts) // 2
        func1 = '\n\n'.join(parts[:mid])
        func2 = '\n\n'.join(parts[mid:])
        return func1, func2
    else:
        # Fallback if there are no blank lines
        return content, content # Very rare error case, shouldn't happen with GPTCloneBench

def main():
    base_dir = "/Users/danielfloresrojas/Documents/Dataset_Java/dataset_clones_java"
    codexglue_dir = os.path.join(base_dir, "tipos_1_2_3_CodeXGLUE")
    metadata_csv = os.path.join(base_dir, "metadata", "clones.csv")
    gpt_dir = os.path.join(base_dir, "tipo_4_GPTCloneBench", "GPTCloneBench", "standalone", "true_semantic_clones", "java")
    out_dir = os.path.join(base_dir, "model_ready")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data.jsonl...")
    id_to_func = {}
    with open(os.path.join(codexglue_dir, "data.jsonl"), "r") as f:
        for line in f:
            obj = json.loads(line)
            id_to_func[str(obj["idx"])] = obj["func"]

    print("Loading pairs from train/valid/test.txt...")
    splits = ["train", "valid", "test"]
    split_pairs = {"train": [], "valid": [], "test": []}
    positive_pairs_to_find = set()

    for s in splits:
        with open(os.path.join(codexglue_dir, f"{s}.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    id1, id2, label = parts[0], parts[1], parts[2]
                    split_pairs[s].append((id1, id2, label))
                    if label == "1":
                        positive_pairs_to_find.add((id1, id2))
                        positive_pairs_to_find.add((id2, id1))

    print("Parsing clones.csv to map types...")
    pair_to_type = {}
    with open(metadata_csv, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                id1, id2 = parts[0], parts[1]
                if (id1, id2) in positive_pairs_to_find:
                    stype = parts[2]
                    sim = parts[4]
                    ctype = get_clone_type(stype, sim)
                    pair_to_type[(id1, id2)] = ctype
                    pair_to_type[(id2, id1)] = ctype

    print("Building CodeXGLUE final objects...")
    final_data = {"train": [], "valid": [], "test": []}
    for s in splits:
        for id1, id2, label in split_pairs[s]:
            func1 = id_to_func.get(id1, "")
            func2 = id_to_func.get(id2, "")
            ctype = "T0" if label == "0" else pair_to_type.get((id1, id2), "UNKNOWN_T123")
            
            final_data[s].append({
                "id1": id1,
                "id2": id2,
                "func1": func1,
                "func2": func2,
                "label": int(label),
                "clone_type": ctype
            })

    print("Processing GPTCloneBench T4...")
    gpt_t4_pairs = []
    gpt_id_counter = 1
    for prompt in ["prompt_1", "prompt_2"]:
        t4_dir = os.path.join(gpt_dir, prompt, "T4")
        if not os.path.exists(t4_dir): continue
        for fname in os.listdir(t4_dir):
            if fname.endswith(".java"):
                with open(os.path.join(t4_dir, fname), "r") as f:
                    content = f.read()
                func1, func2 = split_java_functions(content)
                id1 = f"GPT_T4_{prompt}_{gpt_id_counter}_1"
                id2 = f"GPT_T4_{prompt}_{gpt_id_counter}_2"
                gpt_id_counter += 1
                gpt_t4_pairs.append({
                    "id1": id1,
                    "id2": id2,
                    "func1": func1,
                    "func2": func2,
                    "label": 1,
                    "clone_type": "T4"
                })

    random.shuffle(gpt_t4_pairs)
    n = len(gpt_t4_pairs)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)
    
    final_data["train"].extend(gpt_t4_pairs[:train_end])
    final_data["valid"].extend(gpt_t4_pairs[train_end:valid_end])
    final_data["test"].extend(gpt_t4_pairs[valid_end:])

    print("Writing output JSONL files...")
    for s in splits:
        out_file = os.path.join(out_dir, f"{s}.jsonl")
        random.shuffle(final_data[s]) # shuffle train/valid/test datasets
        with open(out_file, "w") as f:
            for item in final_data[s]:
                f.write(json.dumps(item) + "\n")

    print(f"Done! {len(final_data['train'])} train, {len(final_data['valid'])} valid, {len(final_data['test'])} test pairs.")

if __name__ == "__main__":
    main()
