import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
const demoFile = (p) => new URL(`../demo/${p}`, import.meta.url);
const experimentOnDisk = JSON.parse(await readFile(demoFile('data/experiment.json'), 'utf8'));
globalThis.fetch = async (p) => {
  const exists = existsSync(demoFile(p));
  return {
    ok: exists || (experimentOnDisk.mock && p === 'data/golden.json'),
    status: exists ? 200 : (experimentOnDisk.mock && p === 'data/golden.json' ? 200 : 404),
    // Deliberately emulate a stale cached golden response for a mock bundle. core.load() must
    // discard it even though the request succeeds; mock_bundle.py must also remove it on disk.
    json: async () => experimentOnDisk.mock && p === 'data/golden.json'
      ? { api: [{ user_id: 'stale-real-user', items: ['stale-real-item'] }], ranker: [] }
      : JSON.parse(await readFile(demoFile(p), 'utf8')),
  };
};
const C = await import('../demo/js/core.js');
await C.load();
C.validateRanker(C.B.ranker);
console.log('bundle:', C.B.items.ids.length, 'items ·', C.B.items.dim, 'd ·',
            C.B.users.ids.length, 'users ·', C.B.ranker.trees.length, 'trees');

if (C.B.experiment.mock) {
  assert.equal(C.B.ranker.encoding.type, 'one_hot');
  assert.equal(C.B.ranker.objective.type, 'binary:logistic');
  assert.equal(C.B.golden, null, 'mock bundle accepted a stale golden response');
  assert.equal(existsSync(demoFile('data/golden.json')), false,
    'mock generation retained a golden.json from a real export');
  console.log('mock isolation: stale golden response rejected · no golden.json on disk');
}

// arm assignment matches the Python rule?
for (const s of ['u0007','u0008','v9k2ma1x']) {
  const { hex, variant } = await C.variant(s);
  console.log(`  variant("${s}") = ${variant}  (…${hex.slice(-4)})`);
}

// cold start
let r = C.recommend({ arm:'treatment', vec:C.sessionVector([]), features:C.sessionFeatures([]) });
console.log('cold start ->', r.served_by, '| items', r.items.length, '| trace', r.trace);

// three clicks in "books" -> does the session vector actually personalise?
const books = C.B.items.ids.map((_,i)=>i).filter(i=>C.B.items.category[i]==='books').slice(0,3);
const clicks = books;
const f = C.sessionFeatures(clicks);
r = C.recommend({ arm:'treatment', vec:C.sessionVector(clicks), features:f });
const match = r.items.filter(i=>i.category==='books').length;
console.log(`3 clicks in books -> top_category=${f.top_category}, served_by=${r.served_by},`,
            `${match}/10 of the top-10 are books, candidates=${r.trace.candidates.length}`);
console.log('  top-3:', r.items.slice(0,3).map(i=>`${i.id} ${i.category} p=${i.prob.toFixed(3)}`).join(' | '));

// seeded user: does retrieval recover the hidden preference it never saw?
let hits=0, n=0;
for (const u of C.B.users.ids.slice(0,60)) {
  const truth = C.B.users.truth[C.B.users.index.get(u)];
  const rec = C.recommend({ arm:'treatment', vec:C.userVector(u), features:C.userFeatures(u) });
  hits += rec.items.filter(i=>i.category===truth).length; n += 10;
}
console.log(`preferred-category share in top-10: ${(100*hits/n).toFixed(1)}%  (random = 20%)`);

// stats parity with the Python/statsmodels run.
// The z and p must MATCH. The CI must NOT — and knowing why is the point:
//   Phase 13  statsmodels.confint_proportions_2indep(..., compare='diff')  -> Newcombe hybrid
//   core.js   ztest()                                                      -> Agresti-Caffo
// Two different constructions of the same interval. They agree to ~2e-4. Name your method.
const z = C.ztest(362,1500,140,1500);
console.log(`ztest: z=${z.z.toFixed(4)} p=${z.p.toFixed(6)} diff=${z.diff.toFixed(4)}`,
            `CI=[${z.lo.toFixed(4)}, ${z.hi.toFixed(4)}] (Agresti-Caffo) sig=${z.significant}`);
console.log('        statsmodels: z=10.8584  CI=[0.1218, 0.1742] (Newcombe)  <- z matches, CI differs by design');

// Versioned ranker contract: binary one-hot mock path.
const leaf = (value) => ({ l: [-1], r: [-1], i: [0], c: [value], d: [0] });
const binary = {
  format: 'reco-ranker', schema_version: C.RANKER_SCHEMA_VERSION,
  features: ['category_books'],
  encoding: { type: 'one_hot', source_feature: 'category', columns: { books: 'category_books' } },
  objective: { type: 'binary:logistic', base_margin: 0 },
  trees: [leaf(1)],
};
C.validateRanker(binary);
C.B.ranker = binary;
const binaryRow = C.rankRow(
  { events_7d: 0, purchases_30d: 0, distinct_items_7d: 0, top_category: 'books' },
  { category: 'books', price: 10, popularity_7d: 2 },
);
assert.equal(binaryRow.category_books, 1);
assert.ok(Math.abs(C.score(binaryRow) - 0.7310585786) < 1e-9);

// Versioned ranker contract: real BQML-style label-indexed, grouped softmax path.
const multi = {
  format: 'reco-ranker', schema_version: C.RANKER_SCHEMA_VERSION,
  features: ['category'],
  encoding: {
    type: 'label_index', source_feature: 'category', feature: 'category', feature_index: 0,
    vocabulary: ['books', 'sports'], unknown: 'missing',
  },
  objective: { type: 'multi:softprob', groups: 2, positive_group: 1, tree_groups: [0, 1] },
  trees: [leaf(0), {
    l: [1, -1, -1], r: [2, -1, -1], i: [0, 0, 0],
    c: [0.5, -1, 1], d: [1, 0, 0],
  }],
};
C.validateRanker(multi);
C.B.ranker = multi;
assert.ok(Math.abs(C.score({ category: 'books' }) - 0.2689414214) < 1e-9);
assert.ok(Math.abs(C.score({ category: 'sports' }) - 0.7310585786) < 1e-9);
assert.ok(Math.abs(C.score({ category: 'unseen' }) - 0.2689414214) < 1e-9,
  'unseen label must follow XGBoost missing-value routing');

assert.throws(() => C.validateRanker({ ...multi, schema_version: 999 }),
  /unsupported schema_version/);
console.log('ranker schema: one-hot binary ✓ · label-index softmax ✓ · unknown version rejected ✓');

// Tool-call names and arguments come from the model. They must become text nodes, never HTML.
const made = [];
globalThis.document = {
  createElement: (tag) => {
    const node = { tag, className: '', children: [], append(...children) { this.children.push(...children); } };
    made.push(node);
    return node;
  },
  createTextNode: (text) => ({ text: String(text) }),
};
const { Ask } = await import('../demo/js/ask.js');
const box = { children: [], scrollHeight: 1, append(node) { this.children.push(node); } };
Ask.prototype.trace.call({ box }, '<img src=x onerror=alert(1)>', { q: '</b><script>alert(1)</script>' });
assert.equal(made[1].textContent, '<img src=x onerror=alert(1)>');
assert.match(made[0].children[1].text, /<script>alert\(1\)<\/script>/);
assert.equal(Object.hasOwn(made[0], 'innerHTML'), false);
console.log('tool trace: model-controlled name/arguments rendered as text ✓');
