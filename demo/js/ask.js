/* ask.js — an agent over the recommender, not a chatbot in front of it.
 *
 * The LLM has no access to the catalogue, the vectors or the experiment. It has four TOOLS,
 * and those tools are the same functions the Shop tab calls. So it cannot hallucinate a
 * recommendation: it can only ask the recommender for one and describe what came back.
 *
 * The Groq key lives in a Cloudflare Worker (see worker/index.js). It never touches the
 * browser. If the Worker is not configured, this tab says so plainly instead of pretending.
 */
import * as C from './core.js';

export const WORKER_URL = 'https://reco-ask.reco-ask.workers.dev';   // <- your Worker, e.g. https://reco-ask.<you>.workers.dev

const $ = (s) => document.querySelector(s);

const SYSTEM = `You are the recommender engine at reco-engine-adv, explaining yourself.

Architecture you are part of: user events stream through Pub/Sub and Dataflow into BigQuery;
append-only feature snapshots feed a two-tower TensorFlow Recommenders model; item vectors are
served from Vertex AI Vector Search; a BigQuery ML boosted-tree ranker re-orders the candidates;
a hash-randomised A/B test measures CTR lift over a popular-items baseline.

Rules:
- NEVER invent an item, a score, a variant or a number. Call a tool. If a tool cannot answer it,
  say you cannot answer it.
- Be concise and concrete. Cite the actual numbers the tools return.
- If asked why an item was shown, call explain_item and talk about the features that moved it —
  especially same_category, which is the hidden preference the model learned.
- You may be blunt about limitations: the click model is an assumption, not a measurement.`;

const TOOLS = [
  { type: 'function', function: {
    name: 'get_variant', description: 'Which A/B arm this visitor was hashed into, and why.',
    parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: {
    name: 'recommend',
    description: 'Run the live recommender for this visitor. Returns the ten served items with ranker probabilities, plus whether it was personalised or a cold-start fallback.',
    parameters: { type: 'object', properties: {
      seed_category: { type: 'string', description: 'Optional: seed the session vector with this category, as if the visitor had clicked items in it. One of the catalogue categories.' } } } } },
  { type: 'function', function: {
    name: 'explain_item',
    description: 'The exact ranker feature row and probability for one item, for this visitor.',
    parameters: { type: 'object', required: ['item_id'],
      properties: { item_id: { type: 'string', description: 'e.g. i0314' } } } } },
  { type: 'function', function: {
    name: 'experiment_status',
    description: 'Live impressions, CTR per arm, z, p-value, and the 95% CI on the difference.',
    parameters: { type: 'object', properties: {} } } },
];

export class Ask {
  constructor(state, tally) {
    this.st = state; this.T = tally;
    this.messages = [{ role: 'system', content: SYSTEM }];
    this.box = $('#chat');
    $('#send').addEventListener('click', () => this.send());
    $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') this.send(); });

    if (!WORKER_URL) {
      this.say('system', 'Not wired up yet. Deploy <code>worker/index.js</code> to Cloudflare ' +
        '(free), put your Groq key in it, and set <code>WORKER_URL</code> at the top of ' +
        '<code>js/ask.js</code>. Everything else on this page works without it — the LLM is the ' +
        'narration layer, not the recommender.');
      $('#q').disabled = $('#send').disabled = true;
      return;
    }
    this.say('system', `Ask me why. Try: <em>“why did you show me these ten?”</em> · ` +
      `<em>“what would you show someone who likes books?”</em> · <em>“is the lift significant yet?”</em>`);
  }

  say(who, html) {
    const d = document.createElement('div');
    d.className = 'msg' + (who === 'me' ? ' me' : '');
    d.innerHTML = `<div class="who">${who === 'me' ? 'you' : who}</div><p>${html}</p>`;
    this.box.append(d);
    this.box.scrollTop = this.box.scrollHeight;
    return d;
  }

  trace(name, args) {
    const d = document.createElement('div');
    d.className = 'tool';
    d.innerHTML = `<b>${name}</b>(${Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`;
    this.box.append(d);
    this.box.scrollTop = this.box.scrollHeight;
  }

  /* ── The tools. These are the page's own functions. ─────────────────────── */
  run(name, args) {
    const clicks = this.st.clicks;
    const seeded = (cat) => {
      if (!cat) return clicks;
      const idx = [];
      C.B.items.ids.forEach((_, i) => { if (C.B.items.category[i] === cat && idx.length < 4) idx.push(i); });
      return idx;
    };

    if (name === 'get_variant') {
      return { variant: this.st.arm, sha256_prefix: this.st.hex.slice(0, 12),
        rule: 'int(sha256(session_id),16) % 2 -> 1 = treatment, 0 = control',
        why: 'per-user randomisation; the same visitor always lands in the same arm' };
    }

    if (name === 'recommend') {
      const cl = seeded(args.seed_category);
      const rec = C.recommend({ arm: this.st.arm, vec: C.sessionVector(cl), features: C.sessionFeatures(cl) });
      return {
        variant: rec.variant, served_by: rec.served_by,
        seeded_with: args.seed_category || `${clicks.length} real clicks`,
        items: rec.items.map((i) => ({ id: i.id, category: i.category, price: i.price,
          ranker_prob: i.prob === null ? null : +i.prob.toFixed(4), same_category: i.same_category })),
        note: rec.served_by === 'popular'
          ? 'cold start: no session vector, so the popular-items baseline was served'
          : 'retrieved 50 by exact dot-product NN, then ranked by the boosted tree',
      };
    }

    if (name === 'explain_item') {
      const i = C.B.items.index.get(args.item_id);
      if (i === undefined) return { error: `no item ${args.item_id} in the 500-item catalogue` };
      const uf = C.sessionFeatures(clicks);
      const item = C.itemAt(i);
      const row = C.rankRow(uf, item);
      return { item, your_features: uf, ranker_features: row, ranker_prob: +C.score(row).toFixed(4),
        note: 'same_category = 1 means this item is in the category you have clicked most' };
    }

    if (name === 'experiment_status') {
      const t = this.T.treatment, c = this.T.control;
      const r = C.ztest(t.clicks, t.n, c.clicks, c.n);
      if (!r) return { note: 'no impressions in one of the arms yet — run the simulator in the Lab' };
      return {
        control: { impressions: c.n, clicks: c.clicks, ctr: +r.ctr_c.toFixed(4) },
        treatment: { impressions: t.n, clicks: t.clicks, ctr: +r.ctr_t.toFixed(4), cold_start_fallbacks: t.fallbacks },
        absolute_difference: +r.diff.toFixed(4),
        ci95_on_the_difference: [+r.lo.toFixed(4), +r.hi.toFixed(4)],
        z: +r.z.toFixed(3), p_value: +r.p.toFixed(5), significant_at_005: r.significant,
        caveat: 'this tally is from THIS browser session; the headline number in the README came ' +
                'from 3,000 simulated rounds against the deployed Cloud Run API',
      };
    }
    return { error: 'unknown tool' };
  }

  async send() {
    const q = $('#q').value.trim();
    if (!q) return;
    $('#q').value = '';
    this.say('me', escapeHtml(q));
    this.messages.push({ role: 'user', content: q });

    const thinking = this.say('reco', '…');
    try {
      for (let hop = 0; hop < 4; hop++) {
        const res = await fetch(WORKER_URL, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ messages: this.messages, tools: TOOLS }),
        });
        if (res.status === 429) throw new Error(
          'Groq daily quota is spent. The recommender still works — only the narration is rate-limited.');
        if (!res.ok) throw new Error(`worker returned ${res.status}`);
        const raw = (await res.json()).choices[0].message;
        // Echo back ONLY the fields the API contract defines. GPT-OSS models return an extra
        // `reasoning` field, and blindly pushing the whole object back into the next request
        // is how you get a 400 from a provider that didn't expect its own output.
        const msg = { role: 'assistant', content: raw.content ?? '' };
        if (raw.tool_calls?.length) msg.tool_calls = raw.tool_calls;
        this.messages.push(msg);

        if (msg.tool_calls?.length) {
          thinking.remove();
          for (const call of msg.tool_calls) {
            const args = JSON.parse(call.function.arguments || '{}');
            this.trace(call.function.name, args);
            this.messages.push({ role: 'tool', tool_call_id: call.id,
              content: JSON.stringify(this.run(call.function.name, args)) });
          }
          continue;
        }
        thinking.remove();
        this.say('reco', escapeHtml(msg.content || '(no answer)').replace(/\n/g, '<br>'));
        return;
      }
      thinking.remove();
      this.say('reco', 'Gave up after four tool hops.');
    } catch (e) {
      thinking.remove();
      this.say('system', escapeHtml(e.message));
    }
  }
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));