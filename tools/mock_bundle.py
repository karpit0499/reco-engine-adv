"""mock_bundle.py — a stand-in bundle so demo/ runs before you've touched GCP.

It fabricates the same versioned demo-bundle CONTRACT as the real export. Its ranker declares
the mock-only one-hot/binary scoring path; a real export declares label-index/softmax instead.
The synthetic data has the structure a trained model would have found: five category centroids
in 32-d, items scattered around their
centroid, users near the centroid of the category they secretly prefer, and item norms that
grow with popularity (which is exactly the popularity bias dot-product retrieval really has —
and exactly what the ranker exists to correct).

Nothing here is a claim about your model. It is scaffolding: run tools/export_bundle.py the
moment you have a trained one, and every number on the page becomes real.

    pip install numpy xgboost scikit-learn      # sklearn is for the t-SNE projection
    python tools/mock_bundle.py
    cd demo && python3 -m http.server 8000
"""
import json
import math
import pathlib

import numpy as np
import xgboost as xgb

OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "data"
OUT.mkdir(parents=True, exist_ok=True)
# A golden file is evidence from a particular real cloud export. Keeping it beside freshly
# generated synthetic vectors produces a plausible-looking but meaningless fidelity result.
# Remove it before doing any generation so even a later failure cannot leave a mixed bundle.
(OUT / "golden.json").unlink(missing_ok=True)
# A real export publishes a versioned directory through this atomic pointer.
# Mock generation must opt back into the direct synthetic files or the browser
# would keep loading the prior real bundle despite freshly written mock JSON.
(OUT / "current.json").unlink(missing_ok=True)

CATS = ["electronics", "books", "home", "sports", "beauty"]
N_ITEMS, N_USERS, DIM = 500, 300, 32
rng = np.random.default_rng(7)                      # the generator's seed, for the same reason

item_cat = rng.integers(0, 5, N_ITEMS)
price = rng.uniform(5, 200, N_ITEMS).round(2)
pop = rng.gamma(2.0, 30, N_ITEMS).round().astype(int)
user_pref = rng.integers(0, 5, N_USERS)


def space(seed):
    """One training run: a brand-new, randomly initialised embedding space."""
    r = np.random.default_rng(seed)
    C = r.normal(size=(5, DIM))
    C /= np.linalg.norm(C, axis=1, keepdims=True)          # near-orthogonal in 32-d
    I = C[item_cat] + r.normal(scale=0.34, size=(N_ITEMS, DIM))
    I *= (0.85 + 0.30 * (pop / pop.max()))[:, None]        # popular items get longer vectors
    U = C[user_pref] + r.normal(scale=0.34, size=(N_USERS, DIM))
    return I.astype(np.float32), U.astype(np.float32)


def project2(X):
    """32-d -> 2-d for the constellation. t-SNE, not PCA: five near-orthogonal centroids in
    32-d cannot all be separated by the top two principal components, and two categories end
    up drawn on top of each other. t-SNE is the standard tool for *looking at* an embedding
    space, and the panel says so. Distances on screen are not metric — the maths is 32-d."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=1,
                max_iter=800).fit_transform(X).astype(np.float32)


I1, U1 = space(11)
I2, U2 = space(29)                                          # "run 2" — a different room entirely

flat = lambda A: [round(float(v), 4) for v in A.ravel()]

json.dump({
    "dim": DIM, "categories": CATS,
    "ids": [f"i{n:04d}" for n in range(N_ITEMS)],
    "category": [CATS[c] for c in item_cat],
    "price": [float(p) for p in price],
    "popularity_7d": [int(p) for p in pop],
    "emb": flat(I1), "xy": flat(project2(I1)),
    "emb_v2": flat(I2), "xy_v2": flat(project2(I2)),
}, open(OUT / "items.json", "w"))

# user_features, computed the way Phase 4's SQL computes them
ev = rng.poisson(140, N_USERS) + 5
json.dump({
    "ids": [f"u{n:04d}" for n in range(N_USERS)],
    "truth": [CATS[c] for c in user_pref],
    "emb": flat(U1), "emb_v2": flat(U2),
    "feat": {
        "events_7d": [int(e) for e in ev],
        "purchases_30d": [int(x) for x in rng.poisson(9, N_USERS)],
        "distinct_items_7d": [int(x) for x in np.minimum(ev, rng.poisson(60, N_USERS) + 3)],
        "top_category": [CATS[c] for c in user_pref],       # the 70% preference showing through
    },
}, open(OUT / "users.json", "w"))

# ── the ranker: fit on a rank_training-shaped table, export in the LOSSLESS format ────
rows, labels = [], []
for u in range(N_USERS):
    pos = [i for i in range(N_ITEMS) if (item_cat[i] == user_pref[u]) == (rng.random() < 0.7)][:14]
    neg = rng.integers(0, N_ITEMS, 26)
    for i in list(pos) + list(neg):
        rows.append({
            "events_7d": int(ev[u]), "purchases_30d": 6, "distinct_items_7d": 40,
            "price": float(price[i]), "popularity_7d": int(pop[i]),
            "same_category": int(item_cat[i] == user_pref[u]),
            **{f"category_{c}": int(CATS[item_cat[i]] == c) for c in CATS},
        })
        same = int(item_cat[i] == user_pref[u])
        p_click = 1 / (1 + math.exp(-(-1.7 + 2.1 * same
                                      + 0.006 * pop[i] - 0.005 * price[i])))
        labels.append(int(rng.random() < p_click))

FEATS = sorted(rows[0])
X = np.array([[r[f] for f in FEATS] for r in rows], dtype=np.float32)
clf = xgb.XGBClassifier(n_estimators=40, max_depth=5, learning_rate=0.3,
                        eval_metric="auc", random_state=1)
clf.fit(X, np.array(labels))
b = clf.get_booster()
b.feature_names = FEATS
b.save_model("/tmp/mock.json")
raw = json.load(open("/tmp/mock.json"))
bs = float(str(raw["learner"]["learner_model_param"]["base_score"]).strip("[]"))
# split_conditions are NOT rounded — same reason as export_bundle.py. XGBoost writes them at
# round-trip float32 precision and the browser compares against them with Math.fround().
trees = [{"l": t["left_children"], "r": t["right_children"], "i": t["split_indices"],
          "c": [float(v) for v in t["split_conditions"]], "d": t["default_left"]}
         for t in raw["learner"]["gradient_booster"]["model"]["trees"]]

json.dump({
    "format": "reco-ranker",
    "schema_version": 1,
    "features": FEATS,
    "encoding": {
        "type": "one_hot",
        "source_feature": "category",
        "columns": {c: f"category_{c}" for c in CATS},
    },
    "objective": {
        "type": "binary:logistic",
        "base_margin": math.log(bs / (1 - bs)),
    },
    "trees": trees,
}, open(OUT / "ranker.json", "w"))

kb = sum(p.stat().st_size for p in OUT.glob("*.json")) // 1024
json.dump({"bundle_kb": kb, "mock": True, "evidence": "synthetic_demo_only"},
          open(OUT / "experiment.json", "w"))

print(f"wrote {len(list(OUT.glob('*.json')))} files to {OUT}  ({kb} KB total)")
print("no golden.json — the fidelity check needs the real deployed API (Phase 19).")
