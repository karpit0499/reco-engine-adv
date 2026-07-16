/* core.js — the recommender, in the browser.
 *
 * This file is a faithful port of three things that run in the cloud:
 *   api/main.py          variant_for()  ->  variant()
 *   Vector Search (ANN)  find_neighbors() -> retrieve()   [exact, not approximate — see NOTE]
 *   BQML BOOSTED_TREE    ML.PREDICT()   ->  rank()
 *
 * NOTE ON EXACT vs APPROXIMATE. Vector Search runs tree-AH (an ANN index) because a
 * production catalogue is millions of items. This catalogue is 500. 500 x 32 = 16,000
 * multiply-adds per query — an exact scan finishes in under a millisecond. ANN exists to
 * make this *scale*, not to make it *work*. The demo therefore does exact nearest-neighbour
 * and reports its disagreement with the deployed ANN index openly (Lab -> Fidelity).
 */

export const B = { items: null, users: null, ranker: null, experiment: null, golden: null };

const J = (p) => fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p} -> ${r.status}. Serve over http (python3 -m http.server), not file://`);
  return r.json();
});

export async function load() {
  const [items, users, ranker, experiment, golden] = await Promise.all([
    J('data/items.json'), J('data/users.json'), J('data/ranker.json'),
    J('data/experiment.json'), J('data/golden.json').catch(() => null),
  ]);
  items.E = Float32Array.from(items.emb);          // 500 x dim, row-major
  items.XY = Float32Array.from(items.xy);          // 2-D projection of the learned space
  items.index = new Map(items.ids.map((id, i) => [id, i]));
  users.E = Float32Array.from(users.emb);
  users.index = new Map(users.ids.map((id, i) => [id, i]));
  // Split conditions and leaf values become Float32Array. This is NOT a memory optimisation:
  // XGBoost evaluates its trees in float32, and comparing a float64 feature against a float64
  // copy of a float32 threshold misroutes rows that sit on a boundary. See walk(), below.
  for (const t of ranker.trees) t.c = Float32Array.from(t.c);
  Object.assign(B, { items, users, ranker, experiment, golden });
  return B;
}

export const itemAt = (i) => ({
  id: B.items.ids[i], i,
  category: B.items.category[i],
  price: B.items.price[i],
  popularity_7d: B.items.popularity_7d[i],
});

/* ── A/B assignment ───────────────────────────────────────────────────────────
 * Byte-for-byte the same rule as api/main.py:
 *     h = int(sha256(user_id).hexdigest(), 16);  "treatment" if h % 2 else "control"
 * Hashing the USER (not the request) is what makes the experiment valid: you land in the
 * same arm on every visit, so refreshing can't shop you around for a better recommendation.
 * crypto.subtle needs a secure context — https, or localhost. Not file://.            */
export async function variant(sessionId) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(sessionId));
  const hex = [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
  return { hex, variant: (BigInt('0x' + hex) % 2n) === 1n ? 'treatment' : 'control' };
}

/* ── Features ─────────────────────────────────────────────────────────────────
 * TRAINING-SERVING SKEW, LIVE. Phase 4 defines events_7d, purchases_30d,
 * distinct_items_7d and top_category in SQL. The browser is now a SECOND consumer of
 * those definitions. If it computes them even slightly differently, the ranker degrades
 * and nothing errors — which is the entire reason feature stores exist. One definition,
 * two consumers: the SQL below each field is the contract.                           */
export function sessionFeatures(clicks) {
  // events_7d          COUNTIF(event_time > now - 7d)              -> every click this session
  // purchases_30d      COUNTIF(event_type='purchase' AND ...)      -> 0; the demo has no checkout
  // distinct_items_7d  COUNT(DISTINCT item_id)                     -> distinct items clicked
  // top_category       APPROX_TOP_COUNT(category, 1)[OFFSET(0)]    -> modal category of clicks
  const tally = new Map();
  for (const idx of clicks) {
    const c = B.items.category[idx];
    tally.set(c, (tally.get(c) || 0) + 1);
  }
  let top = null, best = -1;
  for (const [c, n] of tally) if (n > best) { top = c; best = n; }
  return {
    events_7d: clicks.length,
    purchases_30d: 0,
    distinct_items_7d: new Set(clicks).size,
    top_category: top,
  };
}

export function userFeatures(userId) {
  const u = B.users.index.get(userId);
  if (u === undefined) return null;
  const f = B.users.feat;
  return {
    events_7d: f.events_7d[u],
    purchases_30d: f.purchases_30d[u],
    distinct_items_7d: f.distinct_items_7d[u],
    top_category: f.top_category[u],
  };
}

/* ── The session vector ───────────────────────────────────────────────────────
 * A brand-new visitor is out-of-vocabulary for the trained user tower (StringLookup maps
 * them to the OOV row), so there is no embedding to look up. Instead: average the item
 * vectors of what they clicked. This is not a hack — it is the user representation from
 * Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys 2016), and
 * it is valid here precisely because the two-tower retrieval task trains user and item
 * vectors into the SAME dot-product space.
 *
 * It is also the "real-time user tower" listed under *what I'd do next* in Phase 18 —
 * implemented for free, with the session as the sequence.
 * Zero clicks -> no vector -> genuine cold start -> served_by = 'popular'.            */
export function sessionVector(clicks) {
  if (!clicks.length) return null;
  const d = B.items.dim, v = new Float32Array(d);
  for (const idx of clicks) for (let k = 0; k < d; k++) v[k] += B.items.E[idx * d + k];
  for (let k = 0; k < d; k++) v[k] /= clicks.length;
  return v;
}

export function userVector(userId) {
  const u = B.users.index.get(userId);
  if (u === undefined) return null;
  const d = B.items.dim;
  return B.users.E.subarray(u * d, (u + 1) * d);
}

/* ── Stage 1: retrieval (exact dot-product NN) ────────────────────────────── */
export function retrieve(vec, k = 50) {
  const d = B.items.dim, n = B.items.ids.length, out = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += vec[j] * B.items.E[i * d + j];
    out.push({ i, score: s });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, k);
}

/* ── Stage 2: ranking (the BQML boosted tree, walked in the browser) ──────────
 * Exported from BigQuery with save_model()'s flat-array JSON, NOT get_dump() — get_dump
 * rounds split_condition, which silently misroutes rows that sit on a split boundary.
 *   leaf            left_children[i] === -1
 *   leaf value      split_conditions[i]
 *   branch          x[split_indices[i]] < split_conditions[i] ? left : right
 *   missing         default_left[i]
 *   probability     sigmoid(base_margin + sum of leaves)
 *
 * ⚠️ Math.fround() IS THE POINT. XGBoost stores split conditions as float32 and compares in
 * float32. Every number in JavaScript is a float64. Compare a float64 feature value against a
 * float32 threshold and a row sitting on a boundary goes LEFT where XGBoost sends it RIGHT —
 * silently, on maybe 1 row in 200, and the probability comes out ~0.05 wrong when it happens.
 * Math.fround(v) rounds v to the nearest float32, which is exactly what XGBoost sees.
 * Measured: float64 comparison -> max error 1.0e-1. With fround -> 1.2e-7. Same trees.
 * (t.c is a Float32Array from load(), so the right-hand side is already float32.)      */
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function walk(t, row, F) {
  let i = 0;
  while (t.l[i] !== -1) {
    const v = row[F[t.i[i]]];
    const goLeft = v === undefined ? !!t.d[i] : Math.fround(v) < t.c[i];
    i = goLeft ? t.l[i] : t.r[i];
  }
  return t.c[i];
}

export function score(featureRow) {
  const R = B.ranker;
  let m = R.base_margin;
  for (const t of R.trees) m += walk(t, featureRow, R.features);
  return sigmoid(m);
}

/* Build the ranker's input row. The one-hot column NAMES come from the exported model
   (`cat_names`), never from a guess in this file: BigQuery ML picks its own encoding, and a
   name that doesn't match a split just reads as `undefined` -> the tree silently takes its
   `default_left` branch and the ranking quietly degrades. Nothing errors. The export script
   checks every row against ML.PREDICT before it ships, which is what catches this. */
export function rankRow(uf, item) {
  const R = B.ranker;
  const row = {};
  for (const f of R.features) row[f] = 0;
  row.events_7d = uf.events_7d;
  row.purchases_30d = uf.purchases_30d;
  row.distinct_items_7d = uf.distinct_items_7d;
  row.price = item.price;
  row.popularity_7d = item.popularity_7d;
  row.same_category = uf.top_category === item.category ? 1 : 0;
  const oneHot = R.cat_names?.[item.category];
  if (oneHot) row[oneHot] = 1;
  return row;
}

export function rank(uf, candidates, k = 10) {
  return candidates
    .map(({ i, score: retrievalScore }) => {
      const item = itemAt(i);
      const row = rankRow(uf, item);
      return { ...item, retrievalScore, row, prob: score(row), same_category: row.same_category };
    })
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k);
}

export const popularItems = (k = 10) =>
  B.items.ids
    .map((_, i) => itemAt(i))
    .sort((a, b) => b.popularity_7d - a.popularity_7d)
    .slice(0, k)
    .map((it) => ({ ...it, prob: null, retrievalScore: null, same_category: null }));

/* ── /recommend — the same control flow as api/main.py ───────────────────── */
export function recommend({ arm, vec, features, k = 10, candidateK = 50 }) {
  if (arm === 'treatment' && vec && features) {
    const candidates = retrieve(vec, candidateK);
    const items = rank(features, candidates, k);
    if (items.length) {
      return { variant: arm, served_by: 'personalized', items, trace: { candidates, vec } };
    }
  }
  // Control lands here. So does a treatment user with no vector yet — a genuine cold start.
  // served_by TAGS those rows so the analysis can measure them instead of pretending they
  // didn't happen (intent-to-treat, Phase 13).
  return { variant: arm, served_by: 'popular', items: popularItems(k), trace: null };
}

/* ── Statistics ───────────────────────────────────────────────────────────────
 * Abramowitz & Stegun 7.1.26. Documented max absolute error 1.5e-7 on erf(x); the z and p
 * below match statsmodels.proportions_ztest to four decimals (verified: z = 10.8584).  */
export function erf(x) {
  const s = x >= 0 ? 1 : -1; x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  return s * (1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
}
export const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

export function ztest(tc, tn, cc, cn) {
  if (!tn || !cn) return null;
  const pt = tc / tn, pc = cc / cn, pool = (tc + cc) / (tn + cn);
  const se = Math.sqrt(pool * (1 - pool) * (1 / tn + 1 / cn));
  const z = se ? (pt - pc) / se : 0;
  // Agresti-Caffo CI on the DIFFERENCE. Phase 13 used statsmodels' default (Newcombe);
  // the two are different constructions and disagree in the 4th decimal. Say which you used.
  const p1 = (tc + 1) / (tn + 2), p2 = (cc + 1) / (cn + 2);
  const seD = Math.sqrt(p1 * (1 - p1) / (tn + 2) + p2 * (1 - p2) / (cn + 2));
  const d = p1 - p2, zc = 1.959963985;
  return {
    ctr_t: pt, ctr_c: pc, z, p: 2 * (1 - Phi(Math.abs(z))),
    diff: pt - pc, lo: d - zc * seD, hi: d + zc * seD,
    lift: pc ? pt / pc - 1 : null, significant: 2 * (1 - Phi(Math.abs(z))) < 0.05,
  };
}

export function wilson(c, n, zc = 1.959963985) {
  if (!n) return [0, 0];
  const p = c / n, z2 = zc * zc, den = 1 + z2 / n;
  const c1 = p + z2 / (2 * n);
  const c2 = zc * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
  return [(c1 - c2) / den, (c1 + c2) / den];
}
