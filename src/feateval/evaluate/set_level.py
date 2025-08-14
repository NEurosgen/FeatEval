from sklearn.pipeline import Pipeline
from ..preprocess.build import build_preprocessor
from ..metrics.classification import metrics_binary

def eval_feature_set(df, target: str, cols: list[str], cv, estimator_ctor):
    X = df[cols].copy()
    y = df[target].copy()
    scores = []
    for tr, va in cv.split(X, y):
        Xt, yt = X.iloc[tr], y.iloc[tr]
        Xv, yv = X.iloc[va], y.iloc[va]
        est = estimator_ctor()
        pipe = Pipeline([("pre", build_preprocessor(Xt)), ("model", est)])
        pipe.fit(Xt, yt)
        proba = pipe.predict_proba(Xv)
        scores.append(metrics_binary(yv, proba)["pr_auc"])   # базовая метрика MVP
    return scores
