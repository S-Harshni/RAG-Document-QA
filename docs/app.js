'use strict';
// The page runs keyword retrieval (BM25) in the browser over the same passages as the Python code.
// Dense and hybrid retrieval need the embedding model, so their measured results are shown in the table.

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === '') continue;
    if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  e.append(...kids.flat().filter(k => k != null && k !== ''));
  return e;
}
const pct = v => (v * 100).toFixed(1) + '%';
const STOP = new Set('a an and are as at be by for from has have in is it its of on or that the to was were what when where which who whom whose why how with did do does this these those there their than then into'.split(' '));
const tokenize = t => (t.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => !STOP.has(w));

function buildIndex(passages, k1 = 1.5, b = 0.75) {
  const postings = new Map(), lengths = [];
  passages.forEach((p, i) => {
    const counts = new Map(), toks = tokenize(p.t + ' ' + p.x);
    lengths.push(toks.length);
    for (const w of toks) counts.set(w, (counts.get(w) || 0) + 1);
    for (const [w, tf] of counts) { if (!postings.has(w)) postings.set(w, []); postings.get(w).push([i, tf]); }
  });
  const mean = lengths.reduce((a, c) => a + c, 0) / lengths.length;
  const norm = lengths.map(l => k1 * (1 - b + b * l / mean));
  return query => {
    const score = new Float32Array(passages.length);
    for (const w of new Set(tokenize(query))) {
      const list = postings.get(w);
      if (!list) continue;
      const idf = Math.log(1 + (passages.length - list.length + 0.5) / (list.length + 0.5));
      for (const [i, tf] of list) score[i] += idf * tf * (k1 + 1) / (tf + norm[i]);
    }
    return [...score.keys()].filter(i => score[i] > 0).sort((x, y) => score[y] - score[x]).slice(0, 5).map(i => ({ i, score: score[i] }));
  };
}

function bestSentence(question, text) {
  const terms = new Set(tokenize(question));
  let best = '', top = -1;
  for (const s of text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)) {
    const words = new Set(tokenize(s));
    const score = [...terms].filter(t => words.has(t)).length / (1 + 0.1 * Math.sqrt(words.size));
    if (score > top) { best = s.trim(); top = score; }
  }
  return { best, top };
}

(async function main() {
  const D = await (await fetch('data.json')).json();
  const search = buildIndex(D.passages);
  const ds = D.dataset, hybrid = D.retrievers.find(r => r.retriever.startsWith('Hybrid')), bm25 = D.retrievers[0];
  document.getElementById('lede').textContent =
    `Ask a question and get the answer quoted from ${ds.passages.toLocaleString()} passages of ${ds.articles} Wikipedia articles, with the passages it came from. Retrieval is evaluated on ${ds.questions.toLocaleString()} questions.`;
  document.getElementById('foot').textContent = `Data: ${ds.name}, ${ds.licence}. Built by S Harshni.`;

  const result = h('div', {});
  const table = h('table', {},
    h('thead', {}, h('tr', {}, ['Retriever', 'Correct passage ranked 1st', 'In top 5', 'In top 10', 'Mean reciprocal rank'].map((t, i) => h('th', { class: i ? 'num' : '', text: t })))),
    h('tbody', {}, D.retrievers.map(r => h('tr', { class: r === hybrid ? 'best' : '' }, h('td', { text: r.retriever }), h('td', { class: 'num', text: pct(r.recall_at_1) }), h('td', { class: 'num', text: pct(r.recall_at_5) }), h('td', { class: 'num', text: pct(r.recall_at_10) }), h('td', { class: 'num', text: r.mrr.toFixed(3) })))));
  document.getElementById('main').append(
    h('div', { class: 'tiles' }, [
      ['Correct passage in the top 5', pct(hybrid.recall_at_5), `hybrid retrieval; ${pct(bm25.recall_at_5)} for keywords alone`],
      ['Correct passage ranked first', pct(bm25.recall_at_1), 'keyword retrieval, the best at rank 1'],
      ['Answer inside the quoted sentence', pct(D.extractive.answer_in_quoted_sentence), 'extractive mode, no language model'],
      ['Questions evaluated', ds.questions.toLocaleString(), `${ds.passages.toLocaleString()} passages`],
    ].map(([l, v, n]) => h('div', { class: 'tile' }, h('div', { class: 'label', text: l }), h('div', { class: 'value', text: v }), h('div', { class: 'note', text: n })))),
    h('div', { class: 'grid' }, h('figure', { class: 'card wide' }, result),
      h('figure', { class: 'card wide' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Retrieval quality on every question' }),
        h('p', { class: 'sub', text: `Each question has one correct passage. Dense uses ${D.embedding_model.split('/')[1]} embeddings; hybrid merges the two rankings by reciprocal rank fusion.` }))),
        h('div', { class: 'table-box' }, table),
        h('ul', { class: 'note-list' }, [
          'Keywords beat embeddings here because the questions were written by people looking at the passage, so they share its words. Real user questions paraphrase more, which is where embeddings help.',
          'This page runs the keyword retriever in your browser and quotes the best-matching sentence. With an API key, the Python version sends the top passages to a language model that must cite them or decline to answer.',
        ].map(t => h('li', { text: t }))))));

  function ask(question) {
    document.getElementById('q').value = question;
    const hits = search(question);
    if (!hits.length) { result.replaceChildren(h('h3', { text: 'No passage matches that question.' })); return; }
    const scored = hits.map(x => ({ ...x, ...bestSentence(question, D.passages[x.i].x) }));
    const top = scored.reduce((a, c) => c.top > a.top ? c : a, scored[0]);
    result.replaceChildren(
      h('p', { class: 'sub', text: 'Answer (quoted from passage ' + (scored.indexOf(top) + 1) + ')' }), h('p', { class: 'answer', text: top.best }),
      h('p', { class: 'sub', style: 'margin-top:14px', text: 'Passages retrieved' }),
      ...scored.map((x, n) => {
        const p = D.passages[x.i], at = p.x.indexOf(x.best);
        const body = x === top && at >= 0 ? [p.x.slice(0, at), h('mark', { text: x.best }), p.x.slice(at + x.best.length)] : [p.x];
        return h('div', { class: 'passage' }, h('span', { class: 'rank', text: `${n + 1} · score ${x.score.toFixed(1)} · ` }), h('b', { class: 't', text: p.t }), h('div', {}, body));
      }));
  }
  document.getElementById('chips').append(...D.examples.slice(0, 6).map(e => h('button', { type: 'button', text: e.question, onclick: () => ask(e.question) })));
  document.getElementById('ask').addEventListener('submit', e => { e.preventDefault(); const q = document.getElementById('q').value.trim(); if (q) ask(q); });
  ask(D.examples[0].question);

  const btn = document.getElementById('theme');
  const dark = () => (document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  btn.textContent = dark() ? 'Light mode' : 'Dark mode';
  btn.addEventListener('click', () => { document.documentElement.dataset.theme = dark() ? 'light' : 'dark'; btn.textContent = dark() ? 'Light mode' : 'Dark mode'; });
})();
