/* constellation.js — the signature element.
 *
 * Every dot is one of the 500 catalogue items, placed at the 2-D projection of its
 * 32-dimensional vector from the trained item tower. Nothing here is decorative: the
 * clusters you can see ARE the hidden category preferences the two-tower model learned.
 * Colour them by ground-truth category and they snap apart — that is Phase 9's
 * preferred-category metric, made visible instead of printed.
 *
 * Then retrieval ignites the 50 nearest neighbours of your vector, and the ranker
 * numbers the 10 that survive. The funnel, drawn on the model's own map.
 */
import { B } from './core.js';

const NS = 'http://www.w3.org/2000/svg';
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
};

const W = 1000, H = 620, PAD = 34;

export class Constellation {
  constructor(mount) {
    this.svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'sky', role: 'img',
      'aria-label': 'The learned item-embedding space, projected to two dimensions' });
    this.gPlate = el('g', { class: 'plate' });
    this.gLinks = el('g', { class: 'links' });
    this.gDots = el('g', { class: 'dots' });
    this.gMarks = el('g', { class: 'marks' });
    this.svg.append(this.gPlate, this.gLinks, this.gDots, this.gMarks);
    mount.append(this.svg);
    this.dots = [];
    this.colored = false;
  }

  /* Fit the projection to the frame once; the layout never moves again unless the
     embedding space itself is replaced (Lab -> retrain / break). */
  fit(xy) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < xy.length; i += 2) {
      x0 = Math.min(x0, xy[i]); x1 = Math.max(x1, xy[i]);
      y0 = Math.min(y0, xy[i + 1]); y1 = Math.max(y1, xy[i + 1]);
    }
    const sx = (W - 2 * PAD) / (x1 - x0 || 1), sy = (H - 2 * PAD) / (y1 - y0 || 1);
    const s = Math.min(sx, sy);
    const ox = PAD + ((W - 2 * PAD) - (x1 - x0) * s) / 2;
    const oy = PAD + ((H - 2 * PAD) - (y1 - y0) * s) / 2;
    this.pos = (i) => [ox + (xy[i * 2] - x0) * s, oy + (xy[i * 2 + 1] - y0) * s];
  }

  render() {
    this.fit(B.items.XY);
    this.gPlate.replaceChildren();
    for (let r = 90; r <= 460; r += 92) {
      this.gPlate.append(el('circle', { cx: W / 2, cy: H / 2, r, class: 'ring' }));
    }
    this.gDots.replaceChildren();
    this.dots = B.items.ids.map((id, i) => {
      const [x, y] = this.pos(i);
      const c = el('circle', { cx: x, cy: y, r: 3.1, class: 'dot',
        'data-cat': B.items.category[i] });
      c.dataset.i = i;
      this.gDots.append(c);
      return c;
    });
    this.gLinks.replaceChildren();
    this.gMarks.replaceChildren();
  }

  /* Re-place every dot without rebuilding the DOM — used when the Lab swaps the
     embedding space. The dots physically fly to their new coordinates, which is the
     single most useful thing you can show someone about what a retrain actually does. */
  reproject() {
    this.fit(B.items.XY);
    this.dots.forEach((c, i) => {
      const [x, y] = this.pos(i);
      c.setAttribute('cx', x); c.setAttribute('cy', y);
    });
    this.clear();
  }

  colorByCategory(on) {
    this.colored = on;
    this.svg.classList.toggle('by-cat', on);
  }

  clear() {
    this.dots.forEach((d) => {
      d.classList.remove('lit', 'top', 'pop');
      d.style.removeProperty('--d');
    });
    this.gLinks.replaceChildren();
    this.gMarks.replaceChildren();
  }

  /* The orchestrated moment — and it draws BOTH arms, always.
   *
   * Half of your visitors hash into control. If the sky only lit up for treatment, half the
   * people who ever open this page would see nothing happen, which is a strange way to show
   * off a recommender. So: the ten POPULAR items (what control is served) are marked in brass,
   * and the fifty the model RETRIEVES are ignited in ultramarine with the ten it ranks on top.
   *
   * The two arms then have different *shapes*, and that is the whole argument of the project
   * in one image: popularity scatters across the entire space because it ignores you;
   * personalisation collapses onto the region you actually live in. Your arm gets the numerals
   * and the hairlines — the other is drawn, but not claimed.
   */
  show({ vec, candidates, served, popular, arm, marker }) {
    this.clear();

    popular.forEach((i) => this.dots[i].classList.add('pop'));

    if (vec && candidates) {
      candidates.forEach((c, n) => {
        const d = this.dots[c.i];
        d.classList.add('lit');
        d.style.setProperty('--d', `${n * 7}ms`);       // nearest ignites first
      });
      served.forEach(({ i }) => this.dots[i].classList.add('top'));

      const [mx, my] = marker;
      const g = el('g', { class: 'you' });
      g.append(
        el('line', { x1: mx - 13, y1: my, x2: mx + 13, y2: my }),
        el('line', { x1: mx, y1: my - 13, x2: mx, y2: my + 13 }),
        el('circle', { cx: mx, cy: my, r: 7.5, class: 'you-ring' }),
      );
      this.gMarks.append(g);

      if (arm === 'treatment') {
        served.forEach(({ i }, n) => {
          const [x, y] = this.pos(i);
          this.gLinks.append(el('line', { x1: mx, y1: my, x2: x, y2: y,
            class: 'link', style: `--d:${240 + n * 26}ms` }));
          this.gMarks.append(numeral(x, y, n + 1, 240 + n * 26));
        });
      }
    }

    if (arm === 'control') {
      popular.forEach((i, n) => {
        const [x, y] = this.pos(i);
        this.gMarks.append(numeral(x, y, n + 1, 120 + n * 30, 'rank c'));
      });
    }
  }

  /* Where the session vector sits on the map: the mean position of what it retrieved,
     weighted by score. An honest visual — the projection is 2-D, the maths is 32-D. */
  markerFor(candidates) {
    let sx = 0, sy = 0, w = 0;
    const top = candidates.slice(0, 12);
    const min = Math.min(...top.map((c) => c.score));
    for (const c of top) {
      const [x, y] = this.pos(c.i);
      const wi = c.score - min + 0.05;
      sx += x * wi; sy += y * wi; w += wi;
    }
    return [sx / w, sy / w];
  }
}

function numeral(x, y, n, delay, cls = 'rank') {
  const t = el('text', { x: x + 8.5, y: y - 7.5, class: cls, style: `--d:${delay}ms` });
  t.textContent = n;
  return t;
}
