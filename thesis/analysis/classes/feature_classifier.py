#!/usr/bin/env python3
"""feature_classifier.py — the feature-class classifier of the thesis: a gradient-boosted tree model (scikit-learn
HistGradientBoostingClassifier) that maps the token statistics of a feature (tf_feats.py) to one of five classes,
position / token / word / concept / context, with class weights that match the predicted class shares to the judged shares
(prevalence matching, fitted on inner 4-fold out-of-fold probabilities).

The trained model of the thesis is shipped as classifier/feature_classifier.joblib, so classes can be assigned without
any judge labels:

    python thesis/analysis/classes/feature_classifier.py apply 0 62            # work/tf_feats/L<L>.npz -> work/tf_class_v6/L<L>.npz
    python thesis/analysis/classes/feature_classifier.py train work/tf_train.json   # retrain from a training table (tf_train_export.py)

Output per layer: cls (uint8, index into CLASSES, 255 = the feature never fires) and prob (float16, the weighted
probability of the chosen class). Always-on (corpus rate >= .6) and rare (< 30 firings) features also get a class here;
the analyses set them aside as their own bands (bridge_v4.py)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
import json, os, sys
import numpy as np

CLASSES = ["position", "token", "word", "concept", "context"]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "classifier", "feature_classifier.joblib")
D = 262144


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)


def train(table):
    """Fit on a training table {"stats": [...], "rows": [{"x": {stat: value}, "y": class}]}; returns the model dict."""
    import sklearn
    from sklearn.model_selection import StratifiedKFold
    TR = json.load(open(table)) if isinstance(table, str) else table; stats = TR["stats"]; rows = TR["rows"]
    X = np.nan_to_num(np.array([[r["x"].get(s) if r["x"].get(s) is not None else np.nan for s in stats] for r in rows], float), nan=0.0)
    y = np.array([CLASSES.index(r["y"]) for r in rows])
    clf = _gbm().fit(X, y)
    inner = np.zeros((len(y), 5))
    for itr, ite in StratifiedKFold(4, shuffle=True, random_state=1).split(X, y):
        inner[ite] = _gbm().fit(X[itr], y[itr]).predict_proba(X[ite])
    W = np.ones(5); target = np.bincount(y, minlength=5) / len(y)
    for _ in range(50):
        got = np.bincount((inner * W).argmax(1), minlength=5) / len(y); W *= (target / np.maximum(got, 1e-3)) ** 0.5
    print("trained on", len(rows), "judged features; inner-CV accuracy raw", round(float((inner.argmax(1) == y).mean()), 3),
          "prevalence-matched", round(float(((inner * W).argmax(1) == y).mean()), 3), flush=True)
    return {"clf": clf, "W": W, "stats": stats, "classes": CLASSES, "n_train": len(rows), "sklearn": sklearn.__version__}


def save(model, path=MODEL):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True); joblib.dump(model, path, compress=3)


def load(path=MODEL):
    import joblib, sklearn
    m = joblib.load(path)
    if m.get("sklearn") != sklearn.__version__:
        print(f"note: model saved with scikit-learn {m.get('sklearn')}, running {sklearn.__version__}", flush=True)
    return m


def predict(model, z, L):
    """Classes of the fired features of one layer; z = the arrays of tf_feats/L<L>.npz."""
    nf = z["nf"]; idx = np.where(~np.isnan(nf))[0]
    X = np.stack([np.nan_to_num(z[s] if s != "layer" else np.full(D, L, np.float32), nan=0.0)[idx] for s in model["stats"]], 1)
    P = model["clf"].predict_proba(X) * model["W"]; P /= P.sum(1, keepdims=True)
    cls = np.full(D, 255, np.uint8); cls[idx] = P.argmax(1); prob = np.zeros(D, np.float16); prob[idx] = P.max(1)
    return cls, prob


if __name__ == "__main__":
    from paths import WORK
    if sys.argv[1] == "train":
        save(train(sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/tf_train.json")); print("saved", MODEL)
    elif sys.argv[1] == "apply":
        m = load(); out = os.environ.get("OUT", f"{WORK}/tf_class_v6"); os.makedirs(out, exist_ok=True)
        for L in range(int(sys.argv[2]), int(sys.argv[3])):
            z = np.load(f"{WORK}/tf_feats/L{L}.npz", allow_pickle=True); cls, prob = predict(m, z, L)
            np.savez_compressed(f"{out}/L{L}.npz", cls=cls, prob=prob)
            print(L, "classes", np.bincount(cls[cls != 255], minlength=5).tolist(), flush=True)
