/* lab.js — the experiment, the retrain, and the honest fidelity check. */
import * as C from './core.js';

const $ = (s) => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const el = (n, a = {}, txt) => {
  const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]);
  if (txt !== undefined) e.textContent = txt;
  return e;
};
const pct = (x) => (x * 100).toFixed(2) + '%';
const pp = (x) => (x >= 0 ? '+' : '') + (x * 100).toFixed(2) + 'pp';

export class Lab {
  constructor(state, tally, onChange) {
    this.st = state;
    this.T = tally;                    // { control:{n,clicks}, treatment:{n,clicks,fallbacks}, series:[] }
    this.onChange = onChange;

    // Which training run each half of the embedding space currently comes from.
    // 0 = run 1 (the `emb` arrays), 1 = run 2 (the `emb_v2` arrays).
    // Tracked as two independent numbers on purpose: the whole point of the Break button is
    // that they can DISAGREE, and a naive "swap both / swap items" toggle gets this wrong —
    // press Break then Retrain and you'd land on items=run1 + users=run2, which is still
    // broken, while the alarm cheerfully told you Retrain would fix it.
    this.itemV = 0;
    this.userV = 0;
    this.broken = false;

    ['pref', 'boost', 'rounds'].forEach((k) => {
      const s = $('#s-' + k);
      s.addEventListener('input', () => { $('#v-' + k).textContent = k === 'rounds' ? s.value : (+s.value).toFixed(2); });
    });
    $('#run-sim').addEventListener('click', () => this.simulate());
    $('#reset-sim').addEventListener('click', () => this.reset());
    $('#run-fid').addEventListener('click', () => this.fidelity());
    $('#retrain').addEventListener('click', () => this.retrain());
    $('#break').addEventListener('click', () => this.breakIt());

    const hasV2 = !!(C.B.items.emb_v2 && C.B.users.emb_v2);
    $('#retrain').disabled = !hasV2;
    $('#break').disabled = !hasV2;
    if (!hasV2) {
      $('#retrain').title = $('#break').title =
        'Export a second training run (Phase 15) and re-run tools/export_bundle.py to unlock this.';
    }
    this.draw();
  }

  /* ── The Phase-12 click model, in the browser ─────────────────────────────
     It never checks the arm. It only asks how many of the ten match the category the
     simulated user actually likes. Treatment has to *earn* the lift.                */
  simulate() {
    const pref = +$('#s-pref').value;
    const boost = +$('#s-boost').value;
    const rounds = +$('#s-rounds').value;
    const BASE = 0.03, CONV = 0.20;

    const ids = C.B.users.ids;
    let rnd = mulberry32(0xC0FFEE ^ (rounds * 7919));

    for (let r = 0; r < rounds; r++) {
      const u = ids[(rnd() * ids.length) | 0];
      const truth = C.B.users.truth[C.B.users.index.get(u)];
      const arm = variantSync(u);

      // Preference strength is a knob on the *simulated user*, not on the model: it decides
      // how concentrated their taste is, which is what the model had to learn in the first place.
      const uf = C.userFeatures(u);
      const vec = C.userVector(u);
      const useVec = rnd() < pref ? vec : null;      // 1 - pref = users the model never saw enough of
      const rec = C.recommend({ arm, vec: useVec, features: uf });

      const arm_t = this.T[arm];
      arm_t.n++;
      if (rec.served_by === 'popular' && arm === 'treatment') arm_t.fallbacks++;

      const match = rec.items.filter((it) => it.category === truth).length / rec.items.length;
      const pClick = BASE + boost * match;
      if (rnd() < pClick) {
        arm_t.clicks++;
        if (rnd() < CONV) arm_t.convs = (arm_t.convs || 0) + 1;
      }
      if (r % 40 === 0) this.snapshot();
    }
    this.snapshot();
    this.draw();
    this.onChange();
  }

  snapshot() {
    const t = this.T.treatment, c = this.T.control;
    this.T.series.push({
      n: t.n + c.n,
      t: t.n ? t.clicks / t.n : 0,
      c: c.n ? c.clicks / c.n : 0,
    });
    if (this.T.series.length > 240) this.T.series = this.T.series.filter((_, i) => i % 2 === 0);
  }

  reset() {
    Object.assign(this.T, {
      control: { n: 0, clicks: 0, convs: 0 },
      treatment: { n: 0, clicks: 0, convs: 0, fallbacks: 0 },
      series: [],
    });
    this.draw(); this.onChange();
  }

  result() {
    const t = this.T.treatment, c = this.T.control;
    return C.ztest(t.clicks, t.n, c.clicks, c.n);
  }

  draw() {
    const W = 620, H = 240, L = 44, Bm = 26;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart' });
    const s = this.T.series;
    const maxY = Math.max(0.05, ...s.flatMap((p) => [p.t, p.c])) * 1.15;
    const X = (i) => L + (i / Math.max(1, s.length - 1)) * (W - L - 12);
    const Y = (v) => H - Bm - (v / maxY) * (H - Bm - 14);

    for (let g = 0; g <= 4; g++) {
      const y = Y(maxY * g / 4);
      svg.append(el('line', { x1: L, y1: y, x2: W - 12, y2: y, class: 'gl' }));
      svg.append(el('text', { x: 6, y: y + 3.5 }, pct(maxY * g / 4).replace('.00', '')));
    }
    svg.append(el('line', { x1: L, y1: H - Bm, x2: W - 12, y2: H - Bm, class: 'ax' }));

    if (s.length > 1) {
      const path = (k, cls) => el('path', {
        d: s.map((p, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(p[k]).toFixed(1)}`).join(' '),
        class: cls,
      });
      svg.append(path('c', 'lc'), path('t', 'lt'));
      const last = s[s.length - 1];
      svg.append(el('text', { x: W - 60, y: Y(last.t) - 6, style: 'fill:var(--treat)' }, 'treatment'));
      svg.append(el('text', { x: W - 52, y: Y(last.c) + 14, style: 'fill:var(--control)' }, 'control'));
    } else {
      svg.append(el('text', { x: L + 8, y: H / 2 }, 'no impressions yet — run the experiment'));
    }
    svg.append(el('text', { x: L, y: H - 7 }, 'impressions →'));
    $('#chart').replaceChildren(svg);

    const r = this.result();
    const bar = $('#cibar'), mid = $('#cimid'), zero = $('#cizero');
    if (!r) { bar.style.width = '0'; return; }
    // The track spans −10pp … +20pp. Map a difference onto 0–100 % of the track, clamped.
    // (The old version combined this mapping with a `margin-left:50%` on the track and put
    // the zero line in the wrong place — the bar and the dashed line disagreed about where
    // zero was, which on a chart whose ONLY job is "does it clear zero" is not a small bug.)
    const LO = -0.10, HI = 0.20;
    const pos = (v) => Math.max(0, Math.min(100, ((v - LO) / (HI - LO)) * 100));
    const lo = pos(r.lo), hi = pos(r.hi);
    bar.style.left = lo + '%';
    bar.style.width = Math.max(0.6, hi - lo) + '%';
    bar.classList.toggle('crosses', r.lo <= 0 && r.hi >= 0);
    mid.style.left = pos(r.diff) + '%';
    zero.style.left = pos(0) + '%';                       // 33.3 % of the track — the decision line
  }

  /* ── Retrain, and the bug that hides inside it ─────────────────────────────
   * NOTHING is swapped in place. `emb` is always run 1 and `emb_v2` is always run 2; what
   * changes is which one we POINT AT. Mutating the arrays instead is how you end up with a
   * state machine you can't reason about — and this is the one button on the page that has
   * to be exactly right, because its entire job is to demonstrate a subtle bug.          */
  applySpace() {
    const I = C.B.items, U = C.B.users;
    I.E = Float32Array.from(this.itemV ? I.emb_v2 : I.emb);
    I.XY = Float32Array.from(this.itemV ? I.xy_v2 : I.xy);
    U.E = Float32Array.from(this.userV ? U.emb_v2 : U.emb);
  }

  /* A retrain produces both halves of ONE new space. Whatever state you were in — including
     broken — you come out of this consistent. That is what a completed retrain means.    */
  retrain() {
    const target = 1 - this.itemV;
    this.itemV = target;
    this.userV = target;
    this.broken = false;
    this.render(`retrained — both halves are now from run ${target + 1}`);
  }

  /* Break: move the ITEM vectors to the other run and leave the user vectors where they are.
     Press it again and they agree once more, from the other end. */
  breakIt() {
    this.itemV = 1 - this.itemV;
    this.broken = this.itemV !== this.userV;
    this.render();
  }

  /* Preferred-category share, measured at BOTH stages against ground truth the model never
     saw. Measuring only the final top-10 is what lets a broken retriever hide — see below. */
  recall(sample = 150) {
    let r1 = 0, r1n = 0, top = 0, topn = 0;
    for (const u of C.B.users.ids.slice(0, sample)) {
      const truth = C.B.users.truth[C.B.users.index.get(u)];
      const vec = C.userVector(u), uf = C.userFeatures(u);
      if (!vec || !uf) continue;
      const cands = C.retrieve(vec, 50);                      // stage 1, before the ranker
      r1 += cands.filter((c) => C.B.items.category[c.i] === truth).length; r1n += 50;
      const rec = C.recommend({ arm: 'treatment', vec, features: uf });
      top += rec.items.filter((i) => i.category === truth).length; topn += 10;
    }
    return { retrieval: r1n ? 100 * r1 / r1n : 0, served: topn ? 100 * top / topn : 0 };
  }

  render(note) {
    this.applySpace();
    const m = this.recall();
    $('#alarm-slot').innerHTML = this.broken ? `
      <div class="alarm">
        <b>Embedding-space mismatch — and look how well the ranker is hiding it.</b>
        <p>Item vectors are from run ${this.itemV + 1}; user vectors are from run ${this.userV + 1}.
        Each training run initialises a <em>new random space</em>, so these are coordinates in
        different rooms. Retrieval is now doing nothing at all —
        <strong>stage-1 preferred-category share: ${m.retrieval.toFixed(1)}%</strong>, at or
        <em>below</em> the 20% you'd get by picking items with a dice roll.</p>
        <p>And yet the ten items you're served are still
        <strong>${m.served.toFixed(1)}%</strong> on-preference. <b>That gap is the lesson.</b> The
        ranker's strongest feature, <code>same_category</code>, is computed from the
        <em>feature store</em> — <code>top_category</code> and <code>category</code> — not from any
        embedding. So stage 2 never notices that stage 1 broke. It quietly fishes the
        right-category items back out of 50 random candidates and does damage control.</p>
        <p>Nothing errors. Nothing pages. Your headline metric <em>sags</em> instead of cratering,
        which is exactly why this class of bug survives in production. <b>The metric that catches it
        is stage-1 recall, measured before the ranker</b> — the number on the left, not the one on
        the right. This is the <b>Phase 15 consistency window</b>: the minute between
        <code>train.py</code> publishing new user vectors and <code>refresh_index.py</code> pushing
        new item vectors. You close it with blue/green index aliasing. Press <em>Retrain</em> to
        bring both halves onto the same run.</p>
      </div>` : `
      <div class="alarm" style="border-color:var(--rule)">
        <b style="color:var(--ink)">Space is consistent — both halves from run ${this.itemV + 1}.</b>
        <p>${note ? note + ' ' : ''}Stage-1 retrieval: <strong>${m.retrieval.toFixed(1)}%</strong>
        preferred-category in the top 50. Served top-10: <strong>${m.served.toFixed(1)}%</strong>.
        Random baseline is 20%.</p>
      </div>`;
    this.onChange('reproject');
  }

  /* ── Fidelity: does this browser agree with the deployed cloud? ─────────── */
  fidelity() {
    const g = C.B.golden;
    const out = $('#fidelity');
    if (!g) {
      out.innerHTML = `<p class="prose">No golden set in the bundle. Re-run
        <code>tools/export_bundle.py</code> while the Vector Search endpoint is deployed —
        it records what the live API returned so this page can be checked against it.</p>`;
      return;
    }

    let hit = 0, tot = 0;
    for (const row of (g.api || [])) {
      const uf = C.userFeatures(row.user_id);
      const vec = C.userVector(row.user_id);
      if (!uf || !vec) continue;
      const mine = C.recommend({ arm: 'treatment', vec, features: uf }).items.map((i) => i.id);
      const cloud = new Set(row.items);
      hit += mine.filter((i) => cloud.has(i)).length;
      tot += row.items.length;
    }
    let replayWorst = 0, exportedWorst = 0, cloudWorst = 0, rankerRows = 0;
    for (const r of (g.ranker || [])) {
      if (!r?.row || !Number.isFinite(r.browser) || !Number.isFinite(r.ml_predict)) continue;
      const browserNow = C.score(r.row);
      replayWorst = Math.max(replayWorst, Math.abs(browserNow - r.browser));
      exportedWorst = Math.max(exportedWorst, Math.abs(r.browser - r.ml_predict));
      cloudWorst = Math.max(cloudWorst, Math.abs(browserNow - r.ml_predict));
      rankerRows++;
    }

    const agree = tot ? hit / tot : 0;
    out.innerHTML = `
      <div class="bars">
        <div class="bar"><span class="nm">top-10 agreement with the deployed ANN index</span>
          <span class="vl">${pct(agree)}</span>
          <span class="tr"><i style="width:${(agree * 100).toFixed(1)}%"></i></span></div>
        <div class="bar"><span class="nm">max |browser ranker − BQML ML.PREDICT|</span>
          <span class="vl">${rankerRows ? cloudWorst.toExponential(1) : '—'}</span>
          <span class="tr"><i style="width:${rankerRows ? Math.min(100, cloudWorst * 1e7) : 0}%"></i></span></div>
      </div>
      <p class="prose" style="margin:12px 0 0;font-size:12.5px">
        ${rankerRows ? `The ${rankerRows} ranker rows replay the exporter's
        <code>browser</code> values to ${replayWorst.toExponential(1)} max error; the exporter's
        recorded <code>browser</code> ↔ <code>ml_predict</code> residual was
        ${exportedWorst.toExponential(1)}.` : 'No valid ranker golden rows were exported.'}
        Retrieval is <strong>not</strong> exact, and
        that gap is the point: the cloud runs <em>approximate</em> nearest-neighbour (tree-AH) because
        a real catalogue is millions of items. The ${tot - hit} item${tot - hit === 1 ? '' : 's'} of
        ${tot} that differ <em>are</em> the approximation. You are looking at ANN recall loss, priced
        in items.</p>`;
  }
}

/* Same rule as core.variant(), but synchronous — the simulator runs thousands of these and
   crypto.subtle is async. FNV-1a over the id, then parity. This is NOT sha256: it is a
   *simulator* assignment, and its only job is to split users 50/50 and stay stable.
   The real arm — yours — is always the sha256 one. Don't let a convenience hash leak into
   the thing you report. */
function variantSync(id) {
  let h = 0x811c9dc5;
  for (let i = 0; i < id.length; i++) { h ^= id.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0) % 2 ? 'treatment' : 'control';
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
