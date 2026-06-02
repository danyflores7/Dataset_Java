

"""
Inference module for Hybrid TF-IDF + AST + XGBoost model.
"""

import csv
from itertools import combinations
import joblib
import numpy as np
from pathlib import Path

from ast_xgboost_pipeline import (
    extract_ast_features,
    ast_features_to_vector,
    build_pairwise_features,
)

EPSILON = 1e-9


def build_tfidf_pairwise_features(X1_sparse, X2_sparse):
    """
    Same TF-IDF pair feature construction used during training.
    """

    v1 = X1_sparse.toarray().astype(np.float32)
    v2 = X2_sparse.toarray().astype(np.float32)

    abs_diff = np.abs(v1 - v2)
    product = v1 * v2

    dot_product = np.sum(v1 * v2, axis=1)
    norm_1 = np.linalg.norm(v1, axis=1)
    norm_2 = np.linalg.norm(v2, axis=1)

    cosine_similarity = dot_product / (norm_1 * norm_2 + EPSILON)
    cosine_similarity = cosine_similarity.reshape(-1, 1)

    return np.hstack(
        [abs_diff, product, cosine_similarity]
    ).astype(np.float32)


class ClonePredictor:

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent
                / "results"
                / "hybrid_xgboost"
                / "model.joblib"
            )

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.vectorizer = bundle["tfidf_vectorizer"]
        self.label_encoder = bundle["label_encoder"]

    def _extract_hybrid_features(self, code1, code2):



        tfidf_1 = self.vectorizer.transform([code1])
        tfidf_2 = self.vectorizer.transform([code2])

        tfidf_features = build_tfidf_pairwise_features(
            tfidf_1,
            tfidf_2,
        )

      

        ast1 = ast_features_to_vector(
            extract_ast_features(code1)
        )

        ast2 = ast_features_to_vector(
            extract_ast_features(code2)
        )

        ast_features = build_pairwise_features(
            ast1.reshape(1, -1),
            ast2.reshape(1, -1),
        )

      

        hybrid_features = np.hstack(
            [tfidf_features, ast_features]
        ).astype(np.float32)

        return hybrid_features

    def predict(self, code1, code2):

        X = self._extract_hybrid_features(
            code1,
            code2,
        )

        prediction_encoded = self.model.predict(X)[0]

        prediction_label = (
            self.label_encoder.inverse_transform(
                [prediction_encoded]
            )[0]
        )

        probabilities = self.model.predict_proba(X)[0]

        confidence = float(
            np.max(probabilities)
        )

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(
                    self.label_encoder.classes_,
                    probabilities,
                )
            },
        }
    
    def predict_files(self, file1, file2):

        with open(file1, "r", encoding="utf-8") as f:
            code1 = f.read()

        with open(file2, "r", encoding="utf-8") as f:
            code2 = f.read()

        return self.predict(code1, code2)
    

    def analyze_directory(self, directory_path):

        directory = Path(directory_path)

        java_files = list(
            directory.glob("*.java")
        )

        results = []

        for file1, file2 in combinations(java_files, 2):

            prediction = self.predict_files(
                str(file1),
                str(file2)
            )

            results.append({
                "file1": file1.name,
                "file2": file2.name,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"]
            })

        return results
    

    def save_report_csv(self, results, output_file):

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "file1",
                    "file2",
                    "prediction",
                    "confidence"
                ]
            )

            writer.writeheader()    

            for row in results:
                writer.writerow(row)
    

    


if __name__ == "__main__":

    predictor = ClonePredictor()

    results = predictor.analyze_directory(
        "ejemplos"
    )

    for result in results:
        print(result)

    predictor.save_report_csv(
        results,
        "reporte.csv"
    )

    print("\nReporte generado: reporte.csv")