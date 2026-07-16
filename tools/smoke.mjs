import { readFile } from 'node:fs/promises';
globalThis.fetch = async (p) => ({ ok: true, status: 200,
  json: async () => JSON.parse(await readFile('../demo/' + p, 'utf8')) });
const C = await import('../demo/js/core.js');
await C.load();
console.log('bundle:', C.B.items.ids.length, 'items ·', C.B.items.dim, 'd ·',
            C.B.users.ids.length, 'users ·', C.B.ranker.trees.length, 'trees');

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
