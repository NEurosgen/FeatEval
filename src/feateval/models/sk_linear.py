from sklearn.linear_model import LogisticRegression, Ridge

class SKLogReg:
    task = "binary"
    def __init__(self, **kw): self.clf = LogisticRegression(max_iter=1000, **kw)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X): return self.clf.predict(X)
    def predict_proba(self, X): return self.clf.predict_proba(X)[:,1]

class SKRidge:
    task = "regression"
    def __init__(self, **kw): self.clf = Ridge(**kw)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X): return self.clf.predict(X)
