/* shop.js — the interactive surface.
 *
 * Two kinds of click, and the difference matters:
 *   BROWSE click  -> an interaction. Moves your session vector. This is an *event*.
 *   SERVED click  -> feedback on an impression. This is what CTR is measured on.
 * That is the same split the real system makes: reco.events vs reco.impressions/feedback.
 */
import * as C from './core.js';
import { Constellation } from './constellation.js';

const $ = (s) => document.querySelector(s);
const eur = (v) => '€' + v.toFixed(2);

export class Shop {
  constructor(state, onImpression, onClick) {
    this.st = state;                    // { sessionId, arm, hex, clicks: [] }
    this.onImpression = onImpression;
    this.onClick = onClick;
    this.sky = new Constellation($('#sky'));
    this.impression = null;

    this.sky.render();
    $('#bycat').addEventListener('change', (e) => this.sky.colorByCategory(e.target.checked));

    this.renderBrowse();
    this.serve();
  }

  /* A stable, spread-out sample of the catalogue to browse — 24 items across all
     categories, so a visitor can build a preference in three clicks. */
  renderBrowse() {
    const byCat = new Map();
    C.B.items.categories.forEach((c) => byCat.set(c, []));
    C.B.items.ids.forEach((_, i) => byCat.get(C.B.items.category[i])?.push(i));
    // Five items per category, spread through the list rather than taken off the front, so a
    // visitor can build a real preference in three clicks. `arr[r * 7]` can be undefined for a
    // thin category — the guard is why this is an `if` and not a bare push.
    const pick = [];
    for (let r = 0; r < 5; r++) for (const c of byCat.keys()) {
      const arr = byCat.get(c);
      if (arr[r * 7] !== undefined) pick.push(arr[r * 7]);
    }
    this.browsePick = pick.slice(0, 25);
    const rail = $('#browse');
    rail.replaceChildren(...this.browsePick.map((i) => this.tile(C.itemAt(i), {
      onClick: () => this.click(i),
      clicked: () => this.st.clicks.includes(i),
    })));
    this.browseTiles = [...rail.children];
  }

  tile(item, { onClick, score = null, match = null, clicked = () => false }) {
    const b = document.createElement('button');
    b.className = 'tile';
    if (match !== null) b.dataset.match = match;
    if (clicked()) b.classList.add('clicked');
    b.innerHTML =
      `<span class="id">${item.id}</span>` +
      `<span class="cat">${item.category}</span>` +
      `<span class="pr">${eur(item.price)}</span>` +
      (score !== null ? `<span class="sc">${score}</span>` : '');
    if (onClick) b.addEventListener('click', onClick); else b.disabled = true;
    return b;
  }

  click(i) {
    this.st.clicks.push(i);
    this.browseTiles.forEach((t, n) => {
      t.classList.toggle('clicked', this.st.clicks.includes(this.browsePick[n]));
    });
    this.serve();
  }

  /* Click on something we served = feedback on that impression. */
  feedback() {
    if (!this.impression) return;
    this.onClick(this.impression);
    this.impression = null;             // one click per impression; dedupe at the source
    $('#served-note').textContent = 'click logged — see Lab';
  }

  serve() {
    const st = this.st;
    const vec = C.sessionVector(st.clicks);
    const feats = C.sessionFeatures(st.clicks);

    const mine = C.recommend({ arm: st.arm, vec, features: feats });
    const other = C.recommend({ arm: st.arm === 'treatment' ? 'control' : 'treatment', vec, features: feats });

    this.impression = { variant: mine.variant, served_by: mine.served_by, items: mine.items };
    this.onImpression(this.impression, feats.top_category);

    // ── constellation: both arms, always ─────────────────────────────────
    // The model's view is computed even for control visitors — they don't get served it,
    // but they get to SEE it. A demo that shows half its audience an empty screen is a demo
    // that respected the experiment more than the reader.
    const model = st.arm === 'treatment' ? mine : other;
    const popular = C.popularItems(10).map((it) => it.i);
    this.sky.show({
      vec,
      candidates: model.trace?.candidates,
      served: model.trace ? model.items.map((it) => ({ i: it.i })) : [],
      popular,
      arm: st.arm,
      marker: model.trace ? this.sky.markerFor(model.trace.candidates) : null,
    });

    // ── funnel ───────────────────────────────────────────────────────────
    const cold = mine.served_by === 'popular';
    $('#funnel').innerHTML = cold ? `
      <div class="rung"><span class="n">500</span><span class="l">catalogue</span></div>
      <div class="gap">no session vector — this is a cold start, not a bug</div>
      <div class="rung on"><span class="n">10</span><span class="l">most popular, served</span></div>
      <div class="gap">tagged <code>served_by = 'popular'</code> so the experiment can count it
        instead of pretending it didn't happen</div>
    ` : `
      <div class="rung"><span class="n">500</span><span class="l">catalogue</span></div>
      <div class="gap">exact dot-product NN · 16,000 multiply-adds · &lt;1 ms</div>
      <div class="rung on"><span class="n">50</span><span class="l">retrieved</span></div>
      <div class="gap">boosted-tree ranker · ${C.B.ranker.trees.length} trees</div>
      <div class="rung on"><span class="n">10</span><span class="l">served</span></div>`;

    // ── why ──────────────────────────────────────────────────────────────
    const why = $('#why-card');
    if (cold) { why.hidden = true; } else {
      why.hidden = false;
      $('#why').innerHTML = mine.items.map((it) => `
        <div class="bar">
          <span class="nm">${it.id}${it.same_category ? ' · same_category' : ''}</span>
          <span class="vl">${it.prob.toFixed(3)}</span>
          <span class="tr"><i style="width:${(it.prob * 100).toFixed(1)}%"></i></span>
        </div>`).join('');
    }

    // ── rails ────────────────────────────────────────────────────────────
    $('#served').replaceChildren(...mine.items.map((it) => this.tile(it, {
      onClick: () => this.feedback(),
      score: it.prob === null ? null : it.prob.toFixed(2),
      match: it.same_category,
    })));
    $('#counter').replaceChildren(...other.items.map((it) => this.tile(it, {
      onClick: null,
      score: it.prob === null ? null : it.prob.toFixed(2),
      match: it.same_category,
    })));

    $('#served-title').textContent =
      `Served to you — ${mine.variant} / ${mine.served_by}`;
    $('#served-note').textContent = cold
      ? 'click something below and this becomes personalised'
      : 'click one to log a click on this impression';
    $('#browse-note').textContent = st.clicks.length
      ? `${st.clicks.length} click${st.clicks.length > 1 ? 's' : ''} · session vector = mean of ${
          new Set(st.clicks).size} item vectors · top_category = ${feats.top_category}`
      : '0 clicks · no session vector yet · cold start';
  }
}
