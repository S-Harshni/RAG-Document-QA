// The Ask tab runs keyword retrieval (BM25) in the browser over the same chunks as the Python code.
// Dense retrieval and the language models need a model, so their measured results are shown in the other tabs.

const STOP = new Set('a an and are as at be by for from has have in is it its of on or that the to was were what when where which who whom whose why how with did do does this these those there their than then into'.split(' '));
const tokenize = t => (t.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => !STOP.has(w));

function buildIndex(chunks, k1 = 1.5, b = 0.75) {
  const postings = new Map(), lengths = [];
  chunks.forEach((p, i) => {
    const counts = new Map(), toks = tokenize(p.t + ' ' + p.x);
    lengths.push(toks.length);
    for (const w of toks) counts.set(w, (counts.get(w) || 0) + 1);
    for (const [w, tf] of counts) { if (!postings.has(w)) postings.set(w, []); postings.get(w).push([i, tf]); }
  });
  const mean = lengths.reduce((a, c) => a + c, 0) / lengths.length;
  const norm = lengths.map(l => k1 * (1 - b + b * l / mean));
  return query => {
    const score = new Float32Array(chunks.length);
    for (const w of new Set(tokenize(query))) {
      const list = postings.get(w);
      if (!list) continue;
      const idf = Math.log(1 + (chunks.length - list.length + 0.5) / (list.length + 0.5));
      for (const [i, tf] of list) score[i] += idf * tf * (k1 + 1) / (tf + norm[i]);
    }
    return [...score.keys()].filter(i => score[i] > 0).sort((x, y) => score[y] - score[x]).slice(0, 5).map(i => ({ i, score: score[i] }));
  };
}

function bestSentence(question, text) {
  const terms = new Set(tokenize(question));
  let best = '', top = -1;
  for (const sentence of text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)) {
    const words = new Set(tokenize(sentence));
    const score = [...terms].filter(t => words.has(t)).length / (1 + 0.1 * Math.sqrt(words.size));
    if (score > top) { best = sentence.trim(); top = score; }
  }
  return { best, top };
}

const NAMES = { zero_shot: 'Zero-shot', few_shot: 'Few-shot', chain_of_thought: 'Chain of thought' };
const HYBRID = 'Hybrid (rank fusion)', BM25 = 'BM25 (keywords)', DENSE = 'Dense (vector store)';
const pctCol = (label, key) => ({ label, key, num: true, format: F.pct });

const TABS = {
  Ask(D) {
    const search = buildIndex(D.chunks);
    const result = h('div', {});
    const input = h('input', { id: 'q', 'aria-label': 'Question', placeholder: 'Ask a question about the documents', autocomplete: 'off' });
    function ask(question) {
      input.value = question;
      const hits = search(question);
      if (!hits.length) { result.replaceChildren(h('h3', { text: 'No chunk matches that question.' })); return; }
      const scored = hits.map(x => ({ ...x, ...bestSentence(question, D.chunks[x.i].x) }));
      const top = scored.reduce((a, c) => c.top > a.top ? c : a, scored[0]);
      result.replaceChildren(
        h('p', { class: 'sub', text: 'Answer (quoted from chunk ' + (scored.indexOf(top) + 1) + ')' }), h('p', { class: 'answer', text: top.best }),
        h('p', { class: 'sub', style: 'margin-top:14px', text: 'Chunks retrieved' }),
        ...scored.map((x, n) => {
          const p = D.chunks[x.i], at = p.x.indexOf(x.best);
          const body = x === top && at >= 0 ? [p.x.slice(0, at), h('mark', { text: x.best }), p.x.slice(at + x.best.length)] : [p.x];
          return h('div', { class: 'passage' }, h('span', { class: 'rank', text: `${n + 1} · score ${x.score.toFixed(1)} · ` }), h('b', { class: 't', text: p.t }), h('div', {}, body));
        }));
    }
    const form = h('form', { class: 'ask', onsubmit: e => { e.preventDefault(); const q = input.value.trim(); if (q) ask(q); } }, input, h('button', { type: 'submit', text: 'Ask' }));
    ask(D.examples[0]);
    return section(
      intro(`Type a question about any of the ${D.retrieval.dataset.documents} documents. This tab runs the keyword retriever in your browser over the same ${F.int(D.chunks.length)} chunks the Python pipeline uses, and quotes the best-matching sentence. The full pipeline adds a vector store and a language model; their measured results are in the other tabs.`),
      form, h('div', { class: 'chips' }, D.examples.slice(0, 6).map(e => h('button', { type: 'button', text: e, onclick: () => ask(e) }))),
      grid(h('figure', { class: 'card wide' }, result)));
  },

  Retrieval(D) {
    const R = D.retrieval, sizes = R.chunk_sizes, main = sizes.find(x => x.size === R.default_size), vs = R.vector_store;
    const names = [BM25, DENSE, HYBRID], colors = [C.baseline, C.model, C.actual];
    const rows = names.map(n => ({ name: n, ...main.retrievers[n] }));
    return section(
      intro(`Each of the ${F.int(R.dataset.questions)} questions has a known answer span. A retriever is right when a chunk containing that span is in its top results. Chunks are packed from whole sentences with ${F.pct0(R.overlap)} overlap.`),
      tiles([
        ['Right chunk in the top 5', F.pct(main.retrievers[HYBRID].recall_at_5), `hybrid retrieval, ${R.default_size}-word chunks`],
        ['Same, embeddings alone', F.pct(main.retrievers[DENSE].recall_at_5), R.embedding_model.split('/')[1]],
        ['Approximate index vs exact search', F.pct(vs.hnsw_recall_of_exact_top10), 'HNSW finds this share of the exact top 10'],
        ['Questions evaluated', F.int(R.dataset.questions), `${R.dataset.documents} documents, ${F.int(R.dataset.words)} words`],
      ]),
      grid(
        card({
          title: `Retrieval quality with ${R.default_size}-word chunks`, wide: true,
          sub: 'Share of questions whose answer-bearing chunk is in the top k. Hybrid merges the keyword and vector rankings by reciprocal rank fusion.',
          table: { columns: [{ label: 'Retriever', key: 'name' }, pctCol('Ranked 1st', 'recall_at_1'), pctCol('In top 3', 'recall_at_3'), pctCol('In top 5', 'recall_at_5'), pctCol('In top 10', 'recall_at_10'), { label: 'Mean reciprocal rank', key: 'mrr', num: true, format: v => F.dec(v, 3) }], rows, best: r => r.name === HYBRID },
          notes: ['Keywords are strong here because the questions were written by people looking at the text, so they reuse its words. Real users paraphrase more, which is where embeddings earn their place; fusing both is the safe default.'],
        }),
        card({
          title: 'Chunk size: recall in the top 5', sub: 'Larger chunks help keyword search and hurt embeddings, which blur when a chunk covers several topics.',
          legend: names.map((n, i) => ({ name: n, color: colors[i] })),
          table: { columns: [{ label: 'Chunk size (words)', key: 'size', num: true }, { label: 'Chunks', key: 'chunks', num: true, format: F.int }, ...names.map(n => ({ label: n, key: n, num: true, format: F.pct }))], rows: sizes.map(x => ({ size: x.size, chunks: x.chunks, ...Object.fromEntries(names.map(n => [n, x.retrievers[n].recall_at_5])) })) },
        }, p => lineChart(p, { labels: sizes.map(x => x.size), series: names.map((n, i) => ({ name: n, color: colors[i], values: sizes.map(x => x.retrievers[n].recall_at_5) })), yFormat: F.pct0, xFormat: v => v + ' words', label: 'Recall at 5 by chunk size' })),
        card({
          title: 'Chunk size: words sent to the model', sub: 'The cost side of the trade-off: the top 5 chunks have to fit in the context window and are paid for on every question.',
          table: { columns: [{ label: 'Chunk size (words)', key: 'size', num: true }, { label: 'Words in the top 5 chunks', key: 'top5_context_words', num: true, format: F.int }, { label: 'Answer fits inside one chunk', key: 'answer_inside_one_chunk', num: true, format: v => F.pct(v, 2) }], rows: sizes },
        }, p => columnChart(p, { labels: sizes.map(x => x.size + ' words'), values: sizes.map(x => x.top5_context_words), name: 'Words in the top 5 chunks', label: 'Context words by chunk size' })),
        card({
          title: 'Vector store', wide: true, sub: `FAISS index of ${F.int(vs.vectors)} vectors with ${vs.dimensions} dimensions, saved to disk and reloaded.`,
          table: { columns: [{ label: 'Index', key: 'index' }, { label: 'Time per query', key: 'ms', num: true, format: v => F.dec(v, 2) + ' ms' }, { label: 'Share of the exact top 10 found', key: 'recall', num: true, format: v => F.pct(v, 1) }], rows: [{ index: 'Flat (exact search)', ms: vs.flat_ms_per_query, recall: 1 }, { index: 'HNSW (approximate graph search)', ms: vs.hnsw_ms_per_query, recall: vs.hnsw_recall_of_exact_top10 }] },
          notes: ['At this size exact search is already fast, so the approximate index buys nothing yet. It matters at millions of vectors; the test here is that it gives up almost no accuracy.'],
        })));
  },

  'Language models'(D) {
    const L = D.llm, S = L.setup, main = L.models.find(m => m.model === S.main_model), best = [...L.models].sort((a, b) => b.answer_accuracy - a.answer_accuracy)[0];
    const cols = [pctCol('Answer correct', 'answer_accuracy'), pctCol('Citation points to the right chunk', 'citation_accuracy'), pctCol('Declined when the answer was not there', 'correctly_declined'), pctCol('Declined although the answer was there', 'wrongly_declined'), pctCol('Valid JSON', 'valid_json')];
    return section(
      intro(`The top ${S.top_k} chunks go to a language model inside a ${F.int(S.budget_tokens)}-token budget. The model must reply in JSON, cite the chunk it used, and decline when the chunks do not hold the answer. Tested on ${S.answerable} questions whose answer is in the documents and ${S.unanswerable} whose answer is deliberately withheld. All models are open-source and run on a laptop (${S.runtime}).`),
      tiles([
        ['Answers correct', F.pct(best.answer_accuracy), `${best.model}; the right chunk was retrieved for ${F.pct(S.gold_chunk_in_context)}`],
        ['Citations pointing to the right chunk', F.pct(best.citation_accuracy), 'of answers that cite a chunk'],
        ['Declined when the answer was withheld', F.pct(best.correctly_declined), `${S.unanswerable} questions with off-topic chunks`],
        ['Valid structured output', F.pct(best.valid_json), 'replies that parse and pass validation'],
      ]),
      grid(
        card({
          title: 'Three open-source models, same prompt', wide: true, sub: 'Few-shot prompt. An answer is correct when it contains one of the reference answers.',
          table: { columns: [{ label: 'Model', key: 'model' }, ...cols], rows: L.models, best: r => r === best },
        }, p => hbars(p, { rows: L.models.map(m => ({ label: m.model, values: [m.answer_accuracy, m.correctly_declined] })), series: [{ name: 'Answer correct', color: C.actual }, { name: 'Declined when it should', color: C.model }], format: F.pct, max: 1 })),
        card({
          title: `Prompt style (${S.main_model})`, wide: true, sub: 'Zero-shot gives only the rules. Few-shot adds worked examples. Chain of thought asks for the reasoning before the answer.',
          table: { columns: [{ label: 'Prompt', key: 'name' }, ...cols], rows: L.styles.map(x => ({ ...x, name: NAMES[x.style] })) },
        }),
        card({
          title: `Sampling settings (${S.main_model})`, wide: true, sub: 'The same question asked three times with different random seeds, top-p 0.95.',
          table: { columns: [{ label: 'Temperature', key: 'temperature', num: true }, pctCol('Same answer all three times', 'same_answer_all_three_runs'), pctCol('Answer correct', 'answer_accuracy')], rows: L.sampling },
          notes: ['Temperature 0 always picks the likeliest token, so the answer is repeatable. Higher temperatures sample more freely: useful for writing, a liability for factual answers.'],
        })));
  },

  Safety(D) {
    const L = D.llm, I = L.injection, det = L.detector;
    const models = [...new Set(I.map(x => x.model))];
    const get = (m, d) => I.find(x => x.model === m && x.defended === d);
    const rows = models.map(m => ({ model: m, open: get(m, false).attack_success, defended: get(m, true).attack_success, acc_open: get(m, false).answer_accuracy, acc_defended: get(m, true).answer_accuracy }));
    const worst = Math.max(...rows.map(r => r.open)), bestDefended = Math.min(...rows.map(r => r.defended));
    const n = Math.round(1 / Math.min(...I.map(x => x.attack_success).filter(v => v > 0), 1 / 36));
    return section(
      intro('A retrieved document is untrusted input. In this test one of the retrieved chunks is poisoned with an instruction aimed at the model (six attack styles, placed at the start, middle or end). The attack succeeds if the model\'s output contains the planted code word.'),
      tiles([
        ['Attacks that worked, no defence', F.pct(worst), 'the most vulnerable model'],
        ['Attacks that worked, with the defence', F.pct(bestDefended), 'the most robust model'],
        ['Attack strings caught by the detector', F.pct0(det.attacks_flagged), 'a pattern check run before the prompt is built'],
        ['Clean chunks wrongly flagged', F.pct(det.clean_chunks_flagged, 2), `of ${F.int(det.clean_chunks)} chunks`],
      ]),
      grid(
        card({
          title: 'Prompt-injection success rate', wide: true,
          sub: 'The defence wraps each chunk in tags and tells the model that chunk text is reference material, never instructions.',
          legend: [{ name: 'No defence', color: C.model }, { name: 'With the defence', color: C.actual }],
          table: { columns: [{ label: 'Model', key: 'model' }, pctCol('Attack worked, no defence', 'open'), pctCol('Attack worked, with defence', 'defended'), pctCol('Still answered correctly, no defence', 'acc_open'), pctCol('Still answered correctly, with defence', 'acc_defended')], rows },
          notes: ['The defence lowers the success rate; it does not make it zero. That is why the pipeline also checks chunks with a detector, validates the output format and redacts keys, emails and phone numbers before anything is logged.'],
        }, p => hbars(p, { rows: rows.map(r => ({ label: r.model, values: [r.open, r.defended] })), series: [{ name: 'No defence', color: C.model }, { name: 'With the defence', color: C.actual }], format: F.pct, max: Math.max(worst, 0.2) })),
        card({
          title: 'Layers of protection', wide: true,
          table: { columns: [{ label: 'Layer', key: 'layer' }, { label: 'What it does', key: 'what' }], rows: [
            { layer: 'Prompt', what: 'Chunks are wrapped in tags and declared to be reference material, never instructions.' },
            { layer: 'Detector', what: 'A pattern check flags chunks that address the model ("ignore previous instructions", fake system messages, closing tags).' },
            { layer: 'Output validation', what: 'The reply must be JSON with an answer, a citation that exists, and nothing else; anything else is rejected.' },
            { layer: 'Grounding', what: 'The model must cite a chunk or decline, which is how made-up answers are measured on the withheld-answer questions.' },
            { layer: 'Log redaction', what: 'API keys, bearer tokens, emails, phone and card numbers are removed before a prompt or reply is written to a log.' },
          ] },
        })));
  },

  Agent(D) {
    const A = D.llm.agent, S = D.llm.setup;
    return section(
      intro(`An agent may search again with its own keywords before answering. It is tested where single-shot retrieval fails: ${A.questions} questions for which the first search did not return the answer-bearing chunk.`),
      tiles([
        ['Agent answers correct', F.pct(A.agent_accuracy), `single-shot on the same questions: ${F.pct(A.single_shot_accuracy)}`],
        ['Agent found the right chunk', F.pct(A.agent_found_gold_chunk), 'single-shot: 0% by construction'],
        ['Searches per question', F.dec(A.mean_searches, 2), 'a step limit stops endless searching'],
        ['Model', S.main_model, S.runtime],
      ]),
      grid(card({
        title: 'How the agent works', wide: true,
        table: { columns: [{ label: 'Step', key: 'step' }, { label: 'What happens', key: 'what' }], rows: [
          { step: '1. Decide', what: 'Each turn the model returns one JSON action: search with a query of its own, or answer.' },
          { step: '2. Tool call', what: 'A search action runs the hybrid retriever; the chunks go back to the model as the tool result.' },
          { step: '3. Repeat or finish', what: 'The model may rephrase and search again. After four steps without an answer it must decline.' },
        ] },
        notes: ['These are the hardest questions in the set, so the absolute numbers are low. The comparison that matters is agent against single-shot on the same questions.'],
      })));
  },
};

(async function main() {
  const D = await (await fetch('data.json')).json();
  const ds = D.retrieval.dataset;
  if (!D.llm) for (const t of ['Language models', 'Safety', 'Agent']) delete TABS[t];
  document.getElementById('lede').textContent =
    `A retrieval-augmented question-answering pipeline over ${ds.documents} documents (${F.int(ds.words)} words): chunking, embeddings in a FAISS vector store, hybrid retrieval, three open-source language models, prompt-injection tests and a tool-using agent. Every part is measured.`;
  document.getElementById('foot').textContent = `Data: ${ds.name}, ${ds.licence}. Built by S Harshni.`;

  const mainEl = document.getElementById('main'), nav = document.getElementById('tabs');
  const names = Object.keys(TABS), built = {};
  const slug = n => n.toLowerCase().replace(/\s+/g, '-');
  function open(name) {
    hideTip();
    for (const b of nav.children) b.setAttribute('aria-selected', String(b.textContent === name));
    for (const el of mainEl.children) el.hidden = true;
    built[name] ??= mainEl.appendChild(TABS[name](D));
    built[name].hidden = false;
    history.replaceState(null, '', '#' + slug(name));
    renderAll();
  }
  nav.append(...names.map(n => h('button', { type: 'button', role: 'tab', text: n, onclick: () => open(n) })));
  open(names.find(n => '#' + slug(n) === location.hash) || names[0]);

  let timer;
  addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  const btn = document.getElementById('theme');
  const dark = () => (document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  btn.textContent = dark() ? 'Light mode' : 'Dark mode';
  btn.addEventListener('click', () => { document.documentElement.dataset.theme = dark() ? 'light' : 'dark'; btn.textContent = dark() ? 'Light mode' : 'Dark mode'; });
})();
