/* PMC Full-Text Retrieval Tool – front-end (vanilla JS, no dependencies) */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = n => (typeof n === 'number' ? n.toLocaleString() : n);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  };
  const state = { lastQuery: '', lastResults: null, corpus: null, prevTab: 'search' };

  /* ---------------- theme (dark by default) ---------------- */
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    const b = $('#theme-toggle'); if (b) b.textContent = t === 'dark' ? '☀️ Light' : '🌙 Dark';
    try { localStorage.setItem('theme', t); } catch (e) { /* ignore */ }
  }
  let theme = 'dark';
  try { theme = localStorage.getItem('theme') || 'dark'; } catch (e) { /* ignore */ }
  applyTheme(theme);
  $('#theme-toggle').addEventListener('click', () => applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));

  /* ---------------- tabs ---------------- */
  function showTab(name) {
    $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
    if (name === 'corpus' && !state.corpus) loadCorpus();
    window.scrollTo(0, 0);
  }
  $$('.tab').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

  /* ---------------- charts (inline SVG) ---------------- */
  const PALETTE = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2', '#db2777', '#4b5563'];

  function hbarChart(rows, { width = 520, barH = 18, labelW = 130, color = '#93c5fd' } = {}) {
    // rows: [[label, value], ...]
    const max = Math.max(...rows.map(r => r[1]), 1);
    const h = rows.length * (barH + 4) + 4;
    const plotW = width - labelW - 60;
    let s = `<svg viewBox="0 0 ${width} ${h}" width="100%" style="max-width:${width}px;display:block">`;
    rows.forEach(([label, v], i) => {
      const y = i * (barH + 4) + 2, w = Math.max(1, v / max * plotW);
      s += `<text x="${labelW - 6}" y="${y + barH * 0.72}" text-anchor="end" font-size="12" fill="var(--ink)">${esc(label)}</text>`;
      s += `<rect x="${labelW}" y="${y}" width="${w}" height="${barH}" rx="3" fill="${color}"><title>${esc(label)}: ${fmt(v)}</title></rect>`;
      s += `<text x="${labelW + w + 5}" y="${y + barH * 0.72}" font-size="11" fill="var(--muted)">${fmt(v)}</text>`;
    });
    return s + '</svg>';
  }

  function groupedBarChart(groups, series, data, { width = 900, height = 220 } = {}) {
    // groups: x labels (docs), series: legend labels (terms), data[g][s] = value
    const padL = 40, padB = 46, padT = 10, plotW = width - padL - 10, plotH = height - padB - padT;
    const max = Math.max(1, ...data.flat());
    const gw = plotW / Math.max(groups.length, 1), bw = Math.max(2, (gw - 6) / Math.max(series.length, 1));
    let s = `<svg viewBox="0 0 ${width} ${height}" width="100%" style="display:block">`;
    for (let i = 0; i <= 4; i++) {
      const v = max * i / 4, y = padT + plotH - plotH * i / 4;
      s += `<line x1="${padL}" x2="${width - 10}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${padL - 6}" y="${y + 4}" font-size="10" text-anchor="end" fill="var(--muted)">${Math.round(v)}</text>`;
    }
    groups.forEach((g, gi) => {
      const x0 = padL + gi * gw + 3;
      series.forEach((sName, si) => {
        const v = data[gi][si] || 0, h = v / max * plotH;
        s += `<rect x="${x0 + si * bw}" y="${padT + plotH - h}" width="${bw - 1}" height="${h}" fill="${PALETTE[si % PALETTE.length]}"><title>${esc(g)} · ${esc(sName)}: ${v}</title></rect>`;
      });
      s += `<text x="${x0 + gw / 2 - 3}" y="${height - padB + 14}" font-size="10" text-anchor="middle" fill="var(--ink)">${esc(g)}</text>`;
    });
    s += '</svg><div class="legend">' + series.map((n, i) => `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${esc(n)}</span>`).join('') + '</div>';
    return s;
  }

  /* ---------------- search ---------------- */
  async function doSearch(q) {
    q = (q || $('#q').value).trim();
    if (!q) return;
    $('#q').value = q;
    state.lastQuery = q;
    $('#summary').textContent = 'Searching…';
    try {
      const params = new URLSearchParams({ q, method: $('#method').value, op: $('#op').value, limit: $('#limit').value, scope: $('#scope').value });
      const r = await api('/api/search?' + params);
      state.lastResults = r;
      renderAnalysis(r.analysis);
      renderResults(r);
    } catch (e) {
      $('#summary').textContent = 'Error: ' + e.message;
      $('#results').innerHTML = '';
    }
  }

  function renderAnalysis(a) {
    const box = $('#analysis');
    box.classList.remove('hidden');
    let h = '<b>Query analysis:</b> ';
    h += a.tokens.map(t => {
      if (t.role === 'term') return `<span class="chip term">${esc(t.word)} → <b>${esc(t.stem)}</b> <small>df=${t.df}</small></span>`;
      if (t.role === 'stopword') return `<span class="chip stopword" title="stop word – removed">${esc(t.word)}</span>`;
      if (t.role === 'operator') return `<span class="chip operator">${esc(t.word)}</span>`;
      return `<span class="chip field">${esc(t.word)}:</span>`;
    }).join('');
    h += ` <span class="chip">parsed as <code>${esc(a.parsed || '∅')}</code></span>`;
    box.innerHTML = h;
  }

  function renderResults(r) {
    const method = { bm25: 'BM25', tfidf: 'TF-IDF', count: 'match count' }[r.method] || r.method;
    const scopeLabel = { all: 'whole document', abstract: 'abstract only', body: 'body only', title: 'title only' }[r.scope] || r.scope;
    $('#summary').innerHTML = `<b>${r.total}</b> document${r.total === 1 ? '' : 's'} matched in <b>${r.time_ms} ms</b> · ranked by ${method} on <b>${scopeLabel}</b> · showing ${r.results.length}`;
    const box = $('#results');
    if (!r.results.length) {
      box.innerHTML = '<div class="card">No documents match. Try the OR operator, a prefix wildcard (<code>term*</code>) or fewer terms.</div>';
      $('#term-chart').classList.add('hidden');
      return;
    }
    const top = r.results[0].score || 1;
    const ftop = {};
    ['title', 'abstract', 'body'].forEach(f => ftop[f] = Math.max(1e-9, ...r.results.map(d => d.scores[f])));
    box.innerHTML = r.results.map((d, i) => {
      const fieldScores = ['abstract', 'body', 'title'].map(f => {
        const v = d.scores[f], w = v > 0 ? Math.max(3, v / ftop[f] * 100) : 0;
        return `<span class="${r.scope === f ? 'sel' : ''}">${f}</span><i><b style="width:${w}%"></b></i><em>${v}</em>`;
      }).join('');
      const mm = d.match_map, mmax = Math.max(1, ...Object.values(mm).flat());
      const mmRow = f => `<div class="mmap-row"><span class="f-${f}">${f}</span><div class="mmap ${f}" title="${f}: where the matches occur (start → end)">${mm[f].map(v => `<i style="height:${v ? Math.max(14, v / mmax * 100) : 0}%"></i>`).join('')}</div><em><b>${d.field_hits[f] || 0}</b></em></div>`;
      const fs = d.field_stats, ft = d.stats;
      const srow = (k, label) => `<tr><td>${label}</td><td class="f-title">${fmt(fs.title[k])}</td><td class="f-abstract">${fmt(fs.abstract[k])}</td><td class="f-body">${fmt(fs.body[k])}</td><td><b>${fmt(ft[k])}</b></td></tr>`;
      const fcell = (obj, f) => `<em class="f-${f}" title="${f}">${obj[f] || 0}</em>`;
      const hits = Object.entries(d.term_hits).sort((a, b) => b[1] - a[1]).slice(0, 12)
        .map(([t, c]) => { const tf = d.term_field_hits[t] || {}; return `<span class="chip term">${esc(t)} <small class="fsplit">${fcell(tf, 'title')}${fcell(tf, 'abstract')}${fcell(tf, 'body')}</small></span>`; }).join('');
      return `<article class="result">
        <div>
          <div class="rank">#${i + 1} · ${esc(d.doc_id)} · score ${d.score}</div>
          <h3><a href="#" data-doc="${esc(d.doc_id)}">${d.title_html}</a></h3>
          <div class="meta">${esc(d.journal)}${d.year ? ' · ' + esc(d.year) : ''} · ${esc(d.authors.join(', '))}</div>
          <div class="snippet">${d.snippet_html}</div>
          <div class="hits"><span class="legend"><em class="f-title">title</em><em class="f-abstract">abstract</em><em class="f-body">body</em></span>${hits}</div>
        </div>
        <div class="side">
          Relevance score${r.scope !== 'all' ? ` (${esc(r.scope)})` : ''}
          <div class="scorebar"><i style="width:${Math.max(3, d.score / top * 100)}%"></i></div>
          <div class="fieldscores" title="Score computed from the hits inside each field only (length-normalised per field)">${fieldScores}</div>
          <div class="fhits-label">Hits by field · where they occur (start → end) <small>${d.total_hits} total</small></div>
          ${mmRow('title')}${mmRow('abstract')}${mmRow('body')}
          <table class="ministats"><tr><th></th><th class="f-title">title</th><th class="f-abstract">abstract</th><th class="f-body">body</th><th>total</th></tr>
            ${srow('characters', 'chars')}${srow('words', 'words')}${srow('sentences', 'sentences')}${srow('index_terms', 'index terms')}${srow('unique_terms', 'unique terms')}
          </table>
        </div>
      </article>`;
    }).join('');
    $$('a[data-doc]', box).forEach(a => a.addEventListener('click', ev => { ev.preventDefault(); openDoc(a.dataset.doc); }));

    // term frequency chart: top 10 docs × query terms
    const docs = r.results.slice(0, 10);
    const termSet = {};
    docs.forEach(d => Object.entries(d.term_hits).forEach(([t, c]) => termSet[t] = (termSet[t] || 0) + c));
    const terms = Object.entries(termSet).sort((a, b) => b[1] - a[1]).slice(0, 8).map(x => x[0]);
    const data = docs.map(d => terms.map(t => d.term_hits[t] || 0));
    $('#term-chart').classList.remove('hidden');
    $('#term-chart').innerHTML = '<h4>Query-term frequency in the top ' + docs.length + ' documents</h4>' +
      groupedBarChart(docs.map(d => d.doc_id), terms, data);
  }

  $('#search-form').addEventListener('submit', e => { e.preventDefault(); doSearch(); });
  $$('.examples a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); doSearch(a.dataset.q); }));
  ['method', 'op', 'limit', 'scope'].forEach(id => $('#' + id).addEventListener('change', () => state.lastQuery && doSearch(state.lastQuery)));

  /* ---------------- document view ---------------- */
  async function openDoc(id, fromTab) {
    state.prevTab = fromTab || (state.corpus && $('#tab-corpus').classList.contains('active') ? 'corpus' : 'search');
    const params = new URLSearchParams({ q: state.lastQuery, op: $('#op').value });
    let d;
    try { d = await api(`/api/doc/${encodeURIComponent(id)}?${params}`); } catch (e) { alert(e.message); return; }
    const st = d.stats, m = d.meta;
    const tiles = [
      ['characters', 'Characters'], ['characters_no_spaces', 'Characters (no spaces)'], ['words', 'Words'],
      ['unique_words', 'Unique words'], ['sentences', 'Sentences (rule-based)'], ['sentences_naive', 'Sentences (naive . ! ?)'],
      ['paragraphs', 'Body paragraphs'], ['sections', 'Sections'], ['avg_words_per_sentence', 'Avg words / sentence'],
      ['avg_word_length', 'Avg word length'], ['stop_words', 'Stop words'], ['index_terms', 'Index terms (after stop/stem)'],
      ['unique_terms', 'Unique index terms'], ['lexical_diversity', 'Lexical diversity'], ['abstract_words', 'Abstract words'], ['body_words', 'Body words'],
    ];
    let h = `<div class="card doc-head">
      <h2>${esc(m.title)}</h2>
      <div class="meta">${esc(m.journal)}${m.year ? ' · ' + esc(m.year) : ''} · ${esc(m.doc_id)}${m.pmid ? ' · PMID ' + esc(m.pmid) : ''}${m.doi ? ' · <a target="_blank" rel="noopener" href="https://doi.org/' + esc(m.doi) + '">doi:' + esc(m.doi) + '</a>' : ''} · <a target="_blank" rel="noopener" href="https://pmc.ncbi.nlm.nih.gov/articles/${esc(m.doc_id)}/">open in PMC</a></div>
      <div class="meta">${esc(m.authors.join(', '))}</div>
      ${m.keywords.length ? `<div class="hits">${m.keywords.map(k => `<span class="chip">${esc(k)}</span>`).join('')}</div>` : ''}
    </div>`;
    h += `<div class="card"><h2>Document statistics</h2><div class="tiles">` +
      tiles.map(([k, l]) => `<div class="tile${k === 'sentences' ? ' accent' : ''}"><b>${fmt(st[k])}</b><span>${l}</span></div>`).join('') + `</div>`;
    const fs = st.fields;
    const frow = (k, label) => `<tr><td>${label}</td><td class="f-title">${fmt(fs.title[k])}</td><td class="f-abstract">${fmt(fs.abstract[k])}</td><td class="f-body">${fmt(fs.body[k])}</td><td><b>${fmt(st[k])}</b></td></tr>`;
    h += `<h4 class="sub-h">Statistics by field</h4>
      <table class="kv bystats"><tr><th></th><th class="f-title">title</th><th class="f-abstract">abstract</th><th class="f-body">body</th><th>whole document</th></tr>
      ${frow('characters', 'Characters')}${frow('words', 'Words')}${frow('sentences', 'Sentences (rule-based)')}${frow('index_terms', 'Index terms')}${frow('unique_terms', 'Unique index terms')}</table>`;
    if (d.query) {
      h += `<div class="tiles"><div class="tile accent"><b>${d.query_hits}</b><span>highlighted matches for “${esc(d.query)}”</span></div>` +
        Object.entries(d.term_freq_for_query).map(([t, c]) => `<div class="tile"><b>${c}</b><span>tf(${esc(t)})</span></div>`).join('') + '</div>';
    }
    h += `<div class="two"><div><h4 class="sub-h">Top index terms (stemmed)</h4>${hbarChart(st.top_terms)}</div>`;
    h += `<div><h4 class="sub-h">Word distribution by field</h4>${hbarChart([['title', st.title_words], ['abstract', st.abstract_words], ['body', st.body_words]], { color: '#c4b5fd' })}
      <h4 class="sub-h" style="margin-top:14px">Sentence detection: rule-based vs naive</h4>${hbarChart([['rule-based', st.sentences], ['naive', st.sentences_naive]], { color: '#6ee7b7' })}</div></div></div>`;
    h += `<div class="card text" id="doc-text">`;
    d.units.forEach(u => {
      const inner = u.sentences.map(s => `<span class="sent${s.hits ? ' hit' : ''}" data-n="${s.n}">${s.html}</span> `).join('');
      if (u.field === 'title') h += `<h3 style="color:var(--ink);font-size:18px;margin-top:4px">${inner}</h3>`;
      else if (u.field === 'heading') h += `<h3>${inner}</h3>`;
      else if (u.field === 'abstract') h += `<p class="abstract"><b>Abstract.</b> ${inner}</p>`;
      else h += `<p>${inner}</p>`;
    });
    h += '</div>';
    $('#doc').innerHTML = h;
    applyDocToggles();
    showTab('doc');
  }
  function applyDocToggles() {
    const t = $('#doc-text'); if (!t) return;
    t.classList.toggle('eos', $('#show-eos').checked);
    t.classList.toggle('nohl', !$('#show-hl').checked);
    t.classList.toggle('onlyhits', $('#only-hits').checked);
  }
  ['show-eos', 'show-hl', 'only-hits'].forEach(id => $('#' + id).addEventListener('change', applyDocToggles));
  $('#back').addEventListener('click', () => showTab(state.prevTab || 'search'));

  /* ---------------- corpus ---------------- */
  let sortKey = 'doc_id', sortDir = 1;
  async function loadCorpus() {
    $('#corpus').innerHTML = '<div class="card">Loading…</div>';
    try { state.corpus = await api('/api/corpus'); } catch (e) { $('#corpus').innerHTML = '<div class="card">' + esc(e.message) + '</div>'; return; }
    renderCorpus();
  }
  function renderCorpus() {
    const c = state.corpus;
    const tiles = [['documents', 'Documents'], ['characters', 'Characters'], ['words', 'Words'], ['sentences', 'Sentences (rule-based)'],
      ['sentences_naive', 'Sentences (naive)'], ['index_terms', 'Index terms'], ['vocabulary', 'Vocabulary (stems)'],
      ['avg_words_per_doc', 'Avg words / doc'], ['avg_sentences_per_doc', 'Avg sentences / doc'], ['build_time_s', 'Index build time (s)']];
    let h = `<div class="card"><h2>Corpus statistics</h2><div class="tiles">` +
      tiles.map(([k, l]) => `<div class="tile"><b>${fmt(c[k])}</b><span>${l}</span></div>`).join('') + '</div></div>';
    h += `<div class="two"><div class="card"><h2>Most frequent index terms (collection frequency)</h2>${hbarChart(c.top_terms.slice(0, 25))}</div>
          <div class="card"><h2>Terms occurring in most documents (document frequency)</h2>${hbarChart(c.top_df.slice(0, 25), { color: '#c4b5fd' })}</div></div>`;
    h += `<div class="card"><h2>Vocabulary browser</h2><input id="vocab-prefix" placeholder="prefix, e.g. immun"> <button id="vocab-btn">Look up</button><div id="vocab" class="vocab" style="margin-top:10px"></div></div>`;
    const cols = [['doc_id', 'ID'], ['title', 'Title'], ['year', 'Year'], ['characters', 'Chars'], ['words', 'Words'], ['unique_words', 'Unique'],
      ['sentences', 'Sent. (rule)'], ['sentences_naive', 'Sent. (naive)'], ['paragraphs', 'Paras'], ['avg_words_per_sentence', 'W/sent'], ['avg_word_length', 'Avg wlen'], ['unique_terms', 'Terms']];
    const rows = c.documents_list.slice().sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0) * sortDir);
    h += `<div class="card" style="overflow:auto;max-height:70vh"><h2>Documents (${rows.length}) – click a column to sort, a row to open</h2><table class="grid"><thead><tr>` +
      cols.map(([k, l]) => `<th data-k="${k}">${l}${sortKey === k ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map(r => `<tr data-doc="${esc(r.doc_id)}">` + cols.map(([k]) => `<td class="${k === 'title' ? 't' : ''}">${k === 'title' ? esc(r.title) : fmt(r[k])}</td>`).join('') + '</tr>').join('') +
      '</tbody></table></div>';
    $('#corpus').innerHTML = h;
    $$('#corpus th').forEach(th => th.addEventListener('click', () => { if (sortKey === th.dataset.k) sortDir = -sortDir; else { sortKey = th.dataset.k; sortDir = 1; } renderCorpus(); }));
    $$('#corpus tr[data-doc]').forEach(tr => tr.addEventListener('click', () => openDoc(tr.dataset.doc, 'corpus')));
    const vb = async () => {
      const p = $('#vocab-prefix').value.trim();
      const rows = await api('/api/vocab?prefix=' + encodeURIComponent(p) + '&limit=150');
      $('#vocab').innerHTML = rows.length ? rows.map(r => `<span class="chip term" title="df=${r.df}, cf=${r.cf}">${esc(r.term)} <small>${r.cf}/${r.df}</small></span>`).join('') : '<i>no terms</i>';
    };
    $('#vocab-btn').addEventListener('click', vb);
    $('#vocab-prefix').addEventListener('keydown', e => { if (e.key === 'Enter') vb(); });
  }

  /* ---------------- algorithms tab ---------------- */
  const STEM_GOOD = [
    ['immunotherapy', 'immunotherapies', 'immunotherapeutic'],
    ['vaccine', 'vaccines', 'vaccination', 'vaccinated'],
    ['patient', 'patients'], ['probiotic', 'probiotics'], ['infect', 'infected', 'infection', 'infections'],
  ];
  const STEM_BAD = [
    { kind: 'over-stemming', words: ['organization', 'organ'], why: 'organization 經步驟 2 與 4 被砍到 organ，「組織」與「器官」混為一談。' },
    { kind: 'over-stemming', words: ['universal', 'university', 'universe'], why: '三個不同概念都落到同一個詞幹。' },
    { kind: 'over-stemming', words: ['generalized', 'general', 'generation'], why: 'generalized 砍到 gener，和 generation 的詞幹 gener 撞在一起。' },
    { kind: 'over-stemming', words: ['news', 'new'], why: 'news 被當成 new 的複數。' },
    { kind: 'over-stemming', words: ['aids', 'aid', 'aided'], why: 'AIDS 小寫後與「幫助」合併（同形異義）。' },
    { kind: 'under-stemming', words: ['analysis', 'analyses'], why: '單複數變成不同詞幹 analysi / analys，查 analysis 找不到寫 analyses 的文件。' },
    { kind: 'under-stemming', words: ['mouse', 'mice'], why: '不規則複數，純規則無法還原。' },
    { kind: 'under-stemming', words: ['sars', 'sar'], why: 'SARS 的 s 被當複數去掉，萬用字元 sars* 因此找不到。' },
    { kind: 'under-stemming', words: ['policy', 'police'], why: '形近字：polici 與 polic 剛好分開，這次是運氣好；sensitivity 與 sensitive 則都變 sensit。' },
  ];
  let algoLoaded = false;
  async function loadAlgo() {
    if (algoLoaded) return; algoLoaded = true;
    try {
      const sw = await api('/api/stopwords');
      const render = () => {
        const f = $('#sw-filter').value.trim().toLowerCase();
        const chips = (arr, cls) => arr.filter(w => !f || w.includes(f)).map(w => `<span class="chip ${cls}">${esc(w)}</span>`).join('') || '<i>–</i>';
        $('#sw-snowball').innerHTML = chips(sw.snowball, 'sw');
        $('#sw-bio').innerHTML = chips(sw.biomedical, 'sw extra');
        $('#sw-smart').innerHTML = sw.smart.filter(w => !f || w.includes(f)).map(w => `<span class="chip sw ${sw.smart_only.includes(w) ? 'smartonly' : ''}">${esc(w)}</span>`).join('') || '<i>–</i>';
        $('#sw-snowball-h').firstChild.textContent = `Snowball 英文停用詞（${sw.snowball.length} 字） `;
        $('#sw-bio-h').firstChild.textContent = `生醫論文補充（${sw.biomedical.length} 字） `;
        $('#sw-smart-h').firstChild.textContent = `SMART 系統停用詞（Salton, 1971；${sw.smart.length} 字） `;
      };
      const notable = ['new', 'novel', 'use', 'used', 'useful', 'significant', 'different', 'important', 'normal', 'known', 'likely', 'possible', 'various', 'several', 'recent', 'best', 'high', 'low', 'first', 'second', 'one', 'two', 'three', 'zero'].filter(w => sw.smart_only.includes(w));
      $('#sw-smart-note').innerHTML = `SMART 比 Snowball 多 <b>${sw.smart_only.length}</b> 個字（灰底加深者為 Snowball 與生醫補充都沒有的字）。它會移除像 ${notable.map(w => `<code>${esc(w)}</code>`).join('、')} 這類在生醫論文裡<b>帶有訊息</b>的詞，例如「novel biomarker」「significant difference」「low dose」都會被砍掉一半，這是本系統不採用它的原因。反過來，本系統用而 SMART 沒有的只有 ${sw.used_not_in_smart.length} 個：${sw.used_not_in_smart.map(w => `<code>${esc(w)}</code>`).join('、')}。`;
      render(); $('#sw-filter').addEventListener('input', render);
      $('#sw-smart-toggle').addEventListener('click', () => { const el = $('#sw-smart'); el.classList.toggle('hidden'); $('#sw-smart-toggle').textContent = el.classList.contains('hidden') ? '展開' : '收合'; });
    } catch (e) { $('#sw-snowball').textContent = e.message; }
    renderStemGroups($('#stem-good'), STEM_GOOD.map(words => ({ words })));
    renderStemGroups($('#stem-bad'), STEM_BAD);
    runAnalyze(); runStem();
  }
  async function renderStemGroups(box, groups) {
    const all = [...new Set(groups.flatMap(g => g.words))];
    const res = await api('/api/stem?word=' + encodeURIComponent(all.join(',')));
    const stemOf = Object.fromEntries(res.map(r => [r.word, r.stem]));
    box.innerHTML = groups.map(g => {
      const stems = [...new Set(g.words.map(w => stemOf[w]))];
      const merged = stems.length === 1;
      const chips = g.words.map(w => `<span class="chip term">${esc(w)} <small>→ ${esc(stemOf[w])}</small></span>`).join('');
      const tag = g.kind ? `<span class="tag ${g.kind === 'over-stemming' ? 'over' : 'under'}">${g.kind}</span>` : '';
      return `<div class="stem-group">${tag}${chips}<span class="verdict ${merged ? 'ok' : 'no'}">${merged ? '同一詞幹 ' + esc(stems[0]) : '不同詞幹：' + stems.map(esc).join(' / ')}</span>${g.why ? `<div class="why">${esc(g.why)}</div>` : ''}</div>`;
    }).join('');
  }
  async function runStem() {
    const w = $('#stem-word').value.trim(); if (!w) return;
    try {
      const res = await api('/api/stem?word=' + encodeURIComponent(w));
      $('#stem-out').innerHTML = res.map(r => {
        const steps = r.trace.length ? r.trace.map(t => `<tr><td><b>${esc(t.step)}</b></td><td>${esc(t.before)} → <b>${esc(t.after)}</b></td><td class="muted">${esc(t.rule)}</td></tr>`).join('') : '<tr><td colspan="3" class="muted">沒有任何規則套用，原字即詞幹</td></tr>';
        return `<div class="trace"><div class="trace-h"><code>${esc(r.word)}</code> → <code>${esc(r.stem)}</code>${r.stopword ? ' <span class="chip stopword">stop word – 索引時會被移除</span>' : ''}</div><table class="kv">${steps}</table></div>`;
      }).join('');
    } catch (e) { $('#stem-out').textContent = e.message; }
  }
  async function runAnalyze() {
    const text = $('#analyze-text').value.trim(); if (!text) return;
    try {
      const r = await api('/api/analyze?text=' + encodeURIComponent(text));
      const rows = r.tokens.map(t => `<tr class="${t.stopword ? 'sw' : ''}"><td>${t.position}</td><td>${esc(t.text)}</td><td>${esc(t.lower)}</td><td>${t.stem ? '<b>' + esc(t.stem) + '</b>' : '<s>' + esc(t.lower) + '</s>'}</td><td class="muted">${t.stopword ? 'stop word，移除但位置保留' : (t.stem !== t.lower ? 'Porter 詞幹化' : '')}</td></tr>`).join('');
      $('#analyze-out').innerHTML = `<table class="kv tokens"><tr><th>#</th><th>token</th><th>lower</th><th>index term</th><th></th></tr>${rows}</table>
        <p class="hint">句子偵測：規則式 <b>${r.sentences.length}</b> 句、天真法 <b>${r.sentences_naive.length}</b> 句。${r.sentences.map((s_, i) => `<br>${i + 1}. ${esc(s_)}`).join('')}</p>`;
    } catch (e) { $('#analyze-out').textContent = e.message; }
  }
  $('#analyze-btn').addEventListener('click', runAnalyze);
  $('#analyze-text').addEventListener('keydown', e => { if (e.key === 'Enter') runAnalyze(); });
  $('#stem-btn').addEventListener('click', runStem);
  $('#stem-word').addEventListener('keydown', e => { if (e.key === 'Enter') runStem(); });
  $$('.tab').forEach(b => b.addEventListener('click', () => { if (b.dataset.tab === 'algo') loadAlgo(); }));

  /* ---------------- upload / remove ---------------- */
  $('#upload-btn').addEventListener('click', async () => {
    const files = $('#files').files, log = $('#upload-log');
    if (!files.length) { log.textContent = 'Choose at least one XML file first.'; return; }
    log.textContent = '';
    for (const f of files) {
      try {
        const text = await f.text();
        const r = await api('/api/upload?filename=' + encodeURIComponent(f.name), { method: 'POST', headers: { 'Content-Type': 'application/xml' }, body: text });
        log.textContent += `✔ ${f.name}: added ${r.added.join(', ') || '(no article found)'} – index now holds ${r.documents} documents\n`;
      } catch (e) { log.textContent += `✖ ${f.name}: ${e.message}\n`; }
    }
    state.corpus = null;
  });
  const STATUS_LABEL = { added: 'added', updated: 'updated', exists: 'already indexed', error: 'failed' };
  function renderFetchResults(results) {
    const t = $('#fetch-results');
    t.innerHTML = '<tr><th>Input</th><th>PMID</th><th>PMCID</th><th>Status</th><th>Content</th><th>Title / message</th></tr>' + results.map(r => {
      const pmc = r.pmcid ? `<a href="https://pmc.ncbi.nlm.nih.gov/articles/${esc(r.pmcid)}/" target="_blank" rel="noopener">${esc(r.pmcid)}</a>` : '–';
      const pmid = r.pmid ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${esc(r.pmid)}/" target="_blank" rel="noopener">${esc(r.pmid)}</a>` : '–';
      const title = r.doc_id ? `<a href="#" class="doclink" data-doc="${esc(r.doc_id)}">${esc(r.title || r.doc_id)}</a>` : '';
      const note = r.message ? `<div class="muted">${esc(r.message)}</div>` : '';
      const content = r.content ? `<span class="content ${r.content === 'full text' ? 'full' : 'abs'}">${esc(r.content)}</span>` : '';
      return `<tr><td>${esc(r.input)}</td><td>${pmid}</td><td>${pmc}</td><td><span class="status status-${esc(r.status)}">${STATUS_LABEL[r.status] || esc(r.status)}</span></td><td>${content}</td><td>${title}${note}</td></tr>`;
    }).join('');
    t.classList.remove('hidden');
    $$('.doclink', t).forEach(a => a.addEventListener('click', e => { e.preventDefault(); openDoc(a.dataset.doc, 'upload'); }));
  }
  $('#fetch-btn').addEventListener('click', async () => {
    const text = $('#fetch-ids').value.trim(), msg = $('#fetch-msg'), btn = $('#fetch-btn');
    if (!text) { msg.textContent = 'Paste at least one PMID or PMC id first.'; return; }
    btn.disabled = true; msg.textContent = 'Contacting NCBI…';
    try {
      const replace = $('#fetch-replace').checked ? 1 : 0;
      const r = await api('/api/fetch?replace=' + replace, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: text }) });
      const n = s => r.results.filter(x => x.status === s).length;
      msg.textContent = r.results.length
        ? `${n('added')} added, ${n('updated')} updated, ${n('exists')} already indexed, ${n('error')} failed – index now holds ${r.documents} documents`
        : 'No PMID or PMC id found in the text.';
      renderFetchResults(r.results);
      state.corpus = null; loadIndexed();
    } catch (e) { msg.textContent = e.message; }
    finally { btn.disabled = false; }
  });
  $('#fetch-ids').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('#fetch-btn').click(); });

  async function loadIndexed() {
    const t = $('#indexed-list');
    try {
      const c = await api('/api/corpus');
      const docs = c.documents_list || [];
      t.innerHTML = '<tr><th>ID</th><th>PMID</th><th>Year</th><th>Title</th><th>Content</th><th>Words</th><th></th></tr>' + docs.map(d =>
        `<tr><td>${esc(d.doc_id)}</td><td>${esc(d.pmid || '–')}</td><td>${esc(d.year || '')}</td><td><a href="#" class="doclink" data-doc="${esc(d.doc_id)}">${esc(d.title)}</a></td><td><span class="content ${d.content === 'full text' ? 'full' : 'abs'}">${esc(d.content || '')}</span></td><td>${fmt(d.words)}</td><td><button class="small danger" data-remove="${esc(d.doc_id)}">Remove</button></td></tr>`).join('');
      $$('.doclink', t).forEach(a => a.addEventListener('click', e => { e.preventDefault(); openDoc(a.dataset.doc, 'upload'); }));
      $$('button[data-remove]', t).forEach(b => b.addEventListener('click', async () => {
        if (!confirm(`Remove ${b.dataset.remove} from the index?`)) return;
        try { await api('/api/remove?id=' + encodeURIComponent(b.dataset.remove), { method: 'POST' }); state.corpus = null; loadIndexed(); }
        catch (e) { alert(e.message); }
      }));
    } catch (e) { t.innerHTML = `<tr><td>${esc(e.message)}</td></tr>`; }
  }
  $('#indexed-refresh').addEventListener('click', loadIndexed);
  $$('.tab').forEach(b => b.addEventListener('click', () => { if (b.dataset.tab === 'upload') loadIndexed(); }));

  $('#remove-btn').addEventListener('click', async () => {
    const id = $('#remove-id').value.trim(); if (!id) return;
    try { const r = await api('/api/remove?id=' + encodeURIComponent(id), { method: 'POST' }); $('#remove-msg').textContent = `removed – ${r.documents} documents left`; state.corpus = null; loadIndexed(); }
    catch (e) { $('#remove-msg').textContent = e.message; }
  });

  /* ---------------- init ---------------- */
  const initial = new URLSearchParams(location.search).get('q');
  if (initial) doSearch(initial);
})();
