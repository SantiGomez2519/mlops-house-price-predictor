from sklearn.base import BaseEstimator, TransformerMixin


class AddHouseAge(TransformerMixin, BaseEstimator):
    def __init__(self, reference_year=2026, source="year_built"):
        self.reference_year = reference_year
        self.source = source

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["house_age"] = self.reference_year - out[self.source]
        return out.drop(columns=[self.source])
