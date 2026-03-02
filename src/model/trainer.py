import joblib
import numpy as np
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from loguru import logger

from src.model.evaluator import evaluate, EvaluationResult
from src.data.loader import load_dataset
from src.data.preprocessor import TextPreprocessor


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]
        self.preprocessor = TextPreprocessor(
            short_forms=config["preprocessing"].get("short_forms"),
            max_input_length=config["preprocessing"].get("max_input_length", 5000),
        )
        self.vectorizer: TfidfVectorizer | None = None
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.evaluation_results: list[EvaluationResult] = []

    def _build_vectorizer(self) -> TfidfVectorizer:
        tfidf_cfg = self.model_cfg["tfidf"]
        ngram = tuple(tfidf_cfg["ngram_range"])
        return TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 10000),
            ngram_range=ngram,
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
            min_df=tfidf_cfg.get("min_df", 2),
        )

    def _build_candidates(self) -> dict[str, Any]:
        lr_cfg = self.model_cfg["logistic_regression"]
        svm_cfg = self.model_cfg["svm"]
        nb_cfg = self.model_cfg["naive_bayes"]

        return {
            "LogisticRegression": LogisticRegression(
                max_iter=lr_cfg.get("max_iter", 1000),
                class_weight=lr_cfg.get("class_weight", "balanced"),
                C=lr_cfg.get("C", 1.0),
                solver=lr_cfg.get("solver", "lbfgs"),
            ),
            "LinearSVM": SGDClassifier(
                loss="hinge",
                max_iter=svm_cfg.get("max_iter", 1000),
                class_weight=svm_cfg.get("class_weight", "balanced"),
                random_state=self.data_cfg.get("random_state", 42),
            ),
            "NaiveBayes": MultinomialNB(
                alpha=nb_cfg.get("alpha", 0.1),
            ),
        }

    def run(self) -> tuple[Any, TfidfVectorizer, EvaluationResult]:
        """Full training pipeline: load → preprocess → vectorize → train → compare → save."""
        # 1. Load data
        df = load_dataset(self.data_cfg["path"], self.data_cfg.get("encoding", "latin-1"))

        # 2. Clean text
        df["message"] = self.preprocessor.clean_series(df["message"])

        # 3. Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["message"],
            df["label"],
            test_size=self.data_cfg.get("test_size", 0.2),
            random_state=self.data_cfg.get("random_state", 42),
            stratify=df["label"],
        )
        logger.info(f"Train size: {len(X_train_text)} | Test size: {len(X_test_text)}")

        # 4. Vectorize
        self.vectorizer = self._build_vectorizer()
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # 5. Train and compare candidates
        candidates = self._build_candidates()
        best_f1 = -1.0

        for name, model in candidates.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probabilities (NaiveBayes and LR support predict_proba; SVM doesn't by default)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            result = evaluate(name, y_test, y_pred, y_prob)
            self.evaluation_results.append(result)

            if result.f1 > best_f1:
                best_f1 = result.f1
                self.best_model = model
                self.best_model_name = name

        logger.info(f"Best model: {self.best_model_name} (F1={best_f1:.4f})")

        # 6. Cross-validate best model on full data
        logger.info("Running 5-fold cross-validation on best model...")
        self.vectorizer_full = self._build_vectorizer()
        X_full = self.vectorizer_full.fit_transform(df["message"])
        cv_scores = cross_val_score(
            self.best_model, X_full, df["label"], cv=5, scoring="f1"
        )
        logger.info(
            f"CV F1 scores: {cv_scores.round(4)} | Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # 7. Retrain best model on full data
        self.best_model.fit(X_full, df["label"])
        self.vectorizer = self.vectorizer_full
        logger.info("Retrained best model on full dataset")

        # 8. Save
        best_result = next(r for r in self.evaluation_results if r.model_name == self.best_model_name)
        self.save()

        return self.best_model, self.vectorizer, best_result

    def save(self):
        save_dir = Path(self.model_cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.joblib"
        vectorizer_path = save_dir / "vectorizer.joblib"
        meta_path = save_dir / "metadata.joblib"

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump({"model_name": self.best_model_name}, meta_path)

        logger.info(f"Saved model → {model_path}")
        logger.info(f"Saved vectorizer → {vectorizer_path}")
