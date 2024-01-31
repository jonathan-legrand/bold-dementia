from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

class Concatenator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X)

class ListScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        return self

    def transform(self, X, y=None):
        return [self.scaler.fit_transform(X_i) for X_i in X]