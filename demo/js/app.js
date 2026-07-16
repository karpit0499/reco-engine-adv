/* app.js — boot, arm assignment, the readout strip, and the tab router. */
import * as C from './core.js';
import { Shop } from './shop.js';
import { Lab } from './lab.js';
import { Ask } from './ask.js';

const $ = (s) => document.querySelector(s);

/* localStorage, but never fatal — some browsers block it in third-party or private contexts,
   and a portfolio demo that throws on load is worse than one that forgets you. */
const store = (() => {
  const mem = {};
  try { const k = '__t'; localStorage.setItem(k, '1'); localStorage.removeItem(k); return localStorage; }
  catch { return { getItem: (k) => mem[k] ?? null, setItem: (k, v) => { mem[k] = String(v); } }; }
})();

const tally = {
  control: { n: 0, clicks: 0, convs: 0 },
  treatment: { n: 0, clicks: 0, convs: 0, fallbacks: 0 },
  series: [],
};

(async function boot() {
  await C.load();

  let sid = store.getItem('reco.sid');
  if (!sid) {
    sid = 'v' + Math.random().toString(36).slice(2, 10);
    store.setItem('reco.sid', sid);
  }
  const { hex, variant } = await C.variant(sid);
  const state = { sessionId: sid, arm: variant, hex, clicks: [] };

  document.documentElement.style.setProperty(
    '--arm', variant === 'treatment' ? 'var(--treat)' : 'var(--control)');
  // (#arm-word is inside #headline, whose innerHTML is replaced two lines down — so there is
  // deliberately no separate write to it. Setting it here would be dead code.)
  $('#hash-line').textContent =
    `sha256("${sid}") = ${hex.slice(0, 10)}… mod 2 = ${BigInt('0x' + hex) % 2n}`;
  $('#headline').innerHTML = variant === 'treatment'
    ? `You are in the <em>treatment</em> arm. This page is going to try to learn what you like.`
    : `You are in the <em>control</em> arm. You get the most popular items, and nothing else.`;

  // lab must be constructed BEFORE shop: the Shop constructor calls serve()
  // synchronously, which fires onImpression -> lab.snapshot(). If lab isn't
  // defined yet you get "Cannot access 'lab' before initialization" (TDZ).
  // The lab callback references shop, but only fires later (on reproject),
  // by which point shop exists.
  const lab = new Lab(state, tally, (what) => {
    if (what === 'reproject') { shop.sky.reproject(); shop.serve(); }
    readout(); foot();
  });
  const shop = new Shop(state, onImpression, onClick);
  new Ask(state, tally);

  function onImpression(imp) {
    tally[imp.variant].n++;
    if (imp.variant === 'treatment' && imp.served_by === 'popular') tally.treatment.fallbacks++;
    lab.snapshot(); lab.draw(); readout(); foot();
  }
  function onClick(imp) {
    tally[imp.variant].clicks++;
    lab.snapshot(); lab.draw(); readout(); foot();
  }

  /* ── Tabs ─────────────────────────────────────────────────────────────── */
  const tabs = [...document.querySelectorAll('[role=tab]')];
  tabs.forEach((t) => t.addEventListener('click', () => {
    tabs.forEach((o) => {
      const on = o === t;
      o.setAttribute('aria-selected', on);
      $('#' + o.getAttribute('aria-controls')).hidden = !on;
    });
    if (t.id === 't-lab') lab.draw();
  }));

  function readout() {
    const t = tally.treatment, c = tally.control;
    const r = C.ztest(t.clicks, t.n, c.clicks, c.n);
    const E = C.B.experiment || {};
    const cells = [
      ['your arm', state.arm, state.arm === 'treatment' ? 't' : 'c'],
      ['impressions · live', (t.n + c.n).toLocaleString(), ''],
      ['ctr treatment', r ? (r.ctr_t * 100).toFixed(2) + '%' : '—', 't'],
      ['ctr control', r ? (r.ctr_c * 100).toFixed(2) + '%' : '—', 'c'],
      ['p-value · live', r ? (r.p < 1e-4 ? '<0.0001' : r.p.toFixed(4)) : '—', r?.significant ? 'sig' : ''],
      ['shipped lift · 3,000 rounds', E.headline_lift ?? '—', ''],
    ];
    $('#readout').innerHTML = cells.map(([k, v, cls]) =>
      `<div class="cell"><span class="k">${k}</span><span class="v ${cls}">${v}</span></div>`).join('');
  }

  function foot() {
    const E = C.B.experiment || {};
    $('#foot-stats').textContent =
      `${C.B.items.ids.length} items · ${C.B.items.dim}-d · ${C.B.ranker.trees.length} trees · ` +
      `bundle ${E.bundle_kb ?? '?'} KB · exact-NN in the browser`;
  }

  readout(); foot();
})().catch((e) => {
  document.querySelector('.wrap').insertAdjacentHTML('afterbegin',
    `<div class="alarm"><b>Could not load the bundle.</b><p>${e.message}</p></div>`);
});
