"""
TF-IDF embeddings backed by scikit-learn's TfidfVectorizer.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

class Embeddings:
    """Turns documents into L2-normalized TF-IDF vectors."""

    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    @staticmethod
    def require_documents(documents: list[str]) -> None:
        """Raise if `documents` is empty; the single owner of that rule."""
        if not documents:
            raise ValueError("cannot fit on an empty corpus")

    def fit(self, documents: list[str]) -> "Embeddings":
        """Learn the vocabulary and the IDF weight of every term in `documents`."""
        self.require_documents(documents)
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents: list[str]) -> csr_matrix:
        """
        Transform the provided documents into embeddings.
        """
        self.require_documents(documents)

        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: list[str]) -> csr_matrix:
        """Fit the model and transform the documents into embeddings."""
        self.require_documents(documents)
        return self.vectorizer.fit_transform(documents)
