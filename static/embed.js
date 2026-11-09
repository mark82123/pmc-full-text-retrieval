/* Word Embedding retrieval tab (Project #4): Reuters-21578 ModApte dataset, word2vec models trained on
   the train split, word-level nearest neighbours with co-occurrence / topic evidence, the document-level
   retrieval system (TF-IDF vs mean / TF-IDF weighted word vectors vs hybrid), side-by-side comparison,
   evaluation (P@10 / MRR / nDCG@10 by method, query kind and topic) and the research-question summary. */
(() => {
  const { $, $$, esc, fmt, api, showTab, plot, hbarChart } = window.IR;
  const f = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const pct = v => (v == null ? '–' : (v * 100).toFixed(1) + '%');
  const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body == null ? undefined : JSON.stringify(body) });
  const err = (box, e) => { box.innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; };
  const COLORS = { 'tfidf-stem': '#64748b', tfidf: '#2563eb', 'w2v-avg': '#059669', 'w2v-tfidf': '#d97706', hybrid: '#7c3aed' };
  const MODEL_COLORS = ['#0ea5e9', '#f43f5e', '#84cc16', '#a16207', '#be185d', '#0f766e', '#6366f1', '#ea580c'];
  const MEASURES = ['P@5', 'P@10', 'R@10', 'MRR', 'MAP', 'nDCG@10', 'nDCG@20'];
  const st = { info: null, queries: null, evalData: null, neighbours: null, loaded: false, lastSearch: null, pollTimer: null };
  const tchip = t => `<span class="tchip ${esc(t)}">${esc(t)}</span>`;
  const topicChips = ts => (ts && ts.length ? ts.map(tchip).join('') : '<span class="muted">no topic</span>');
  const gradeTag = g => g == null ? '' : `<span class="grade g${g}">${g === 2 ? 'rel 2' : g === 1 ? 'rel 1' : 'not rel'}</span>`;
  const methodLabel = m => { const x = st.info && st.info.methods.find(y => y.method === m); return x ? `${x.letter} ${x.method}` : m; };
  const colorOf = m => COLORS[m] || '#64748b';
  const modelColor = name => MODEL_COLORS[Math.max(0, (st.info ? st.info.models.findIndex(m => m.name === name) : 0)) % MODEL_COLORS.length];
  const profileBar = prof => `<span class="profile" title="${esc(prof.topics.map(t => `${t.topic} ${pct(t.share)}`).join(', '))}">${prof.topics.map(t => `<i style="width:${Math.round(t.share * 100)}%;background:${{ earn: '#86efac', acq: '#93c5fd', crude: '#fca5a5', grain: '#fcd34d', trade: '#c4b5fd' }[t.topic] || '#ccc'}"></i>`).join('')}</span>`;

  /* ------------------------------------------------------------------ *
   * 1. dataset
   * ------------------------------------------------------------------ */
  async function loadInfo() {
    const box = $('#emb-info');
    try {
      const i = await api('/api/reuters/info');
      st.info = i;
      const ds = i.dataset, te = i.test, cov = i.coverage;
      box.innerHTML = `<div class="tiles kv-tiles">
        <div class="tile"><b>${fmt(ds.documents)}</b><span>ModApte documents · train ${fmt(ds.splits.train)} / test ${fmt(ds.splits.test)} · ${ds.topics_total} distinct topics · ${fmt(ds.without_body)} BRIEF without body</span></div>
        <div class="tile accent"><b>${fmt(te.documents)}</b><span>test documents = search collection · ${fmt(te.tokens)} word tokens → ${fmt(te.index_terms)} terms · ${te.avg_length} per document (${te.min_length}–${te.max_length})</span></div>
        <div class="tile"><b>${fmt(te.vocabulary)}</b><span>test vocabulary (surface words) · ${fmt(te.hapax_df)} with df = 1</span></div>
        <div class="tile"><b>${cov ? fmt(cov.model_vocabulary) : '–'}</b><span>embedding vocabulary (min_count 5, train split)${cov ? ` · covers ${pct(cov.types_share)} of the test word types, ${pct(cov.tokens_share)} of the tokens` : ''}</span></div>
        <div class="tile"><b>${i.queries}</b><span>topic queries · 4 kinds × 5 topics · relevance from TOPICS</span></div>
        <div class="tile"><b>${i.build_time_s} s</b><span>test vector space build time (pure Python)</span></div>
      </div>
      <div class="two">
        <div><h4>推薦的五個主題（文件數）</h4><table class="grid num"><tr><th>topic</th><th>name</th><th>all</th><th>train</th><th>test</th><th>test, only topic (grade 2)</th></tr>
          ${ds.rows.map(r => `<tr><td>${tchip(r.topic)}</td><td>${esc(r.name)} <span class="muted">${esc(r.name_zh)}</span></td><td>${fmt(r.total)}</td><td>${fmt(r.train)}</td><td>${fmt(r.test)}</td><td>${fmt(r.test_only)}</td></tr>`).join('')}</table>
          <p class="hint" style="margin-top:6px">grain 新聞幾乎都同時帶 wheat / corn 等子主題，所以 grade 2（唯一主題）很少；earn 新聞幾乎都只有 earn 一個主題。</p></div>
        <div><h4>每篇文件的主題數</h4>${hbarChart(ds.topics_per_document.slice(0, 7).map(x => [x.topics + ' topic' + (x.topics === 1 ? '' : 's'), x.documents]), { width: 420, barH: 14, labelW: 90 })}
          <h4 style="margin-top:10px">test 集合 df 最高的詞（idf 最低）</h4><div class="chips">${i.test_top_df.map(r => `<span class="chip" title="cf ${fmt(r.cf)}">${esc(r.term)} <small>df ${fmt(r.df)}</small></span>`).join('')}</div>
          <p class="hint" style="margin-top:6px">said、mln、dlrs、pct、vs 出現在大半新聞中：TF-IDF 給它們極低權重，而「詞向量平均」（方法 C）對每個詞一視同仁——這正是方法 B 用 TF-IDF 加權的理由。</p></div>
      </div>`;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 2. models + training
   * ------------------------------------------------------------------ */
  function renderModels(models, active) {
    const box = $('#emb-models');
    if (!models.length) { box.innerHTML = '<p class="muted">尚無模型。用下方表單訓練，或在終端機執行 <code>python3 cli.py reuters train --model sg --dim 100 --epochs 5</code>。</p>'; return; }
    box.innerHTML = `<div class="tscroll"><table class="grid num"><tr><th></th><th>model</th><th>architecture</th><th>dim</th><th>window</th><th>min_count</th><th>epochs</th><th>vocabulary</th><th>final loss</th><th>train time</th><th>coverage (types / tokens)</th><th>trained</th><th></th></tr>
      ${models.map(m => `<tr class="${m.active ? 'sel' : ''}"><td><i class="dot" style="background:${modelColor(m.name)};width:10px;height:10px;border-radius:50%;display:inline-block"></i></td><td><b>${esc(m.name)}</b>${m.active ? ' <span class="grade g2">active</span>' : ''}</td><td>${esc(m.model)}</td><td>${m.dim}</td><td>${m.window}</td><td>${m.min_count}</td><td>${m.epochs}</td><td>${fmt(m.vocabulary)}</td><td>${f(m.final_loss, 3)}</td><td>${m.train_seconds ? Math.round(m.train_seconds / 60) + ' min' : '–'}</td><td>${m.coverage ? pct(m.coverage.types_share) + ' / ' + pct(m.coverage.tokens_share) : '<span class="muted">(after first use)</span>'}</td><td>${esc(m.trained || '')}</td><td>${m.active ? '' : `<button class="small" data-use="${esc(m.name)}">use</button>`}</td></tr>`).join('')}</table></div>
      <p class="hint" style="margin-top:6px">loss = 每個訓練樣本的平均 negative-sampling loss（skip-gram 與 CBOW 的樣本定義不同，loss 不能跨架構比較）。所有模型都在同一份 train 切分（9,603 篇、約 68K 句）上訓練，min_count 5 → 9,996 個詞。</p>`;
    $$('button[data-use]', box).forEach(b => b.addEventListener('click', async () => { await post('/api/reuters/select?name=' + encodeURIComponent(b.dataset.use)); await refreshModels(); }));
    fillModelPickers(models, active);
  }

  function fillModelPickers(models, active) {
    const sel = $('#eq-model');
    sel.innerHTML = models.map(m => `<option value="${esc(m.name)}"${m.name === active ? ' selected' : ''}>${esc(m.name)}</option>`).join('') || '<option value="">(no model)</option>';
    const chips = (id, on) => {
      const box = $('#' + id);
      box.innerHTML = models.map(m => `<label class="${on.includes(m.name) ? 'on' : ''}"><input type="checkbox" value="${esc(m.name)}"${on.includes(m.name) ? ' checked' : ''}><i class="dot" style="background:${modelColor(m.name)};width:10px;height:10px;border-radius:50%;display:inline-block"></i>${esc(m.name)}</label>`).join('');
      $$('input', box).forEach(i => i.addEventListener('change', () => i.parentElement.classList.toggle('on', i.checked)));
    };
    const sg = models.find(m => m.model === 'skip-gram' && m.dim === 100 && m.window === 5) || models.find(m => m.model === 'skip-gram');
    const cbow = models.find(m => m.model === 'cbow' && m.dim === 100 && m.window === 5 && m.epochs === 5) || models.find(m => m.model === 'cbow');
    const pair = [sg, cbow].filter(Boolean).map(m => m.name);
    const on = pair.length ? pair : (active ? [active] : []);
    if (!$('#en-models').children.length) { chips('en-models', on); chips('ee-models', on); }
  }

  async function refreshModels() {
    try {
      const r = await api('/api/reuters/models');
      if (st.info) st.info.models = r.models;
      renderModels(r.models, r.active);
      renderJobs(r.jobs);
      if (r.jobs.some(j => j.state === 'queued' || j.state === 'running')) {
        clearTimeout(st.pollTimer); st.pollTimer = setTimeout(refreshModels, 3000);
      }
    } catch (e) { err($('#emb-models'), e); }
  }

  function renderJobs(jobs) {
    const box = $('#emb-jobs');
    const mine = jobs.filter(j => j.corpus === 'reuters');
    if (!mine.length) { box.innerHTML = ''; return; }
    box.innerHTML = mine.slice(-6).reverse().map(j => `<div class="note-box" style="margin-top:8px"><b>${esc(j.name)}</b> · ${esc(j.state)}${j.epoch ? ` · epoch ${j.epoch}` : ''}${j.loss != null ? ` · loss ${f(j.loss, 3)}` : ''}${j.elapsed ? ` · ${Math.round(j.elapsed)} s` : ''}${j.message ? ` · <span class="muted">${esc(j.message)}</span>` : ''}
      <div class="scorebar big"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div></div>`).join('');
  }

  async function train() {
    const body = { model: $('#et-arch').value, dim: +$('#et-dim').value, window: +$('#et-window').value, min_count: +$('#et-min').value, epochs: +$('#et-epochs').value, seed: +$('#et-seed').value };
    try { await post('/api/reuters/train', body); await refreshModels(); } catch (e) { err($('#emb-jobs'), e); }
  }

  /* ------------------------------------------------------------------ *
   * 3. word level
   * ------------------------------------------------------------------ */
  const picked = id => $$('#' + id + ' input:checked').map(i => i.value);
  const setPicked = (id, list) => $$('#' + id + ' input').forEach(i => { i.checked = list.includes(i.value); i.parentElement.classList.toggle('on', i.checked); });

  function tagsFor(nb, wordTop) {
    const tags = [];
    if (nb.both >= 10 && nb.pmi != null && nb.pmi >= 1.5) tags.push('<span class="tag ctx" title="often in the same stories">情境共現</span>');
    if (nb.count < 30) tags.push('<span class="tag rare" title="few training occurrences">稀有</span>');
    if (wordTop && nb.top_topic && nb.top_topic === wordTop) tags.push('<span class="tag topic" title="same dominant topic">同主題</span>');
    return tags.join('');
  }

  async function neighbours() {
    const box = $('#en-out');
    const models = picked('en-models');
    if (!models.length) { box.innerHTML = '<p class="muted">choose at least one model</p>'; return; }
    box.innerHTML = '<p class="muted">computing…</p>';
    try {
      const r = await api('/api/reuters/similar?' + new URLSearchParams({ words: $('#en-words').value, models: models.join(','), n: $('#en-n').value }));
      st.neighbours = r;
      let h = `<p class="hint">cos = cosine of the word vectors; 同篇 = documents of the train split that contain both words; PMI = log( P(both) / (P(a)·P(b)) ) over documents; 主題 = the topic most of the test documents containing the neighbour carry. The coloured bar is the topic profile of the query word in the test split (earn green, acq blue, crude red, grain yellow, trade purple).</p>`;
      h += '<div class="vcols">' + r.words.map(w => {
        const prof = w.topic_profile;
        const top = prof.topics.reduce((a, b) => (b.share > (a ? a.share : 0) ? b : a), null);
        const wordTop = top && top.documents ? top.topic : '';
        return `<div class="vcol"><h4>${esc(w.word)} <span class="muted">df train ${fmt(w.df_train)} · test ${fmt(prof.df)}</span></h4>${profileBar(prof)}
          ${w.models.map(mr => mr.in_model ? `<div style="margin-top:6px"><span style="color:${modelColor(mr.model)};font-weight:600">${esc(mr.model)}</span> <span class="muted">count ${fmt(mr.count)}</span>
            <div class="tscroll"><table class="grid num nbtab"><tr><th>neighbour</th><th>cos</th><th>count</th><th>同篇</th><th>PMI</th><th>主題</th></tr>
            ${mr.neighbours.map(nb => `<tr><td class="w">${esc(nb.word)}${tagsFor(nb, wordTop)}</td><td>${f(nb.similarity, 3)}</td><td>${fmt(nb.count)}</td><td>${fmt(nb.both)}</td><td>${nb.pmi == null ? '–' : f(nb.pmi, 1)}</td><td style="text-align:left">${nb.top_topic ? `${tchip(nb.top_topic)}<small class="muted">${pct(nb.top_topic_share)}</small>` : '<span class="muted">–</span>'}</td></tr>`).join('')}</table></div></div>`
            : `<div style="margin-top:6px"><span style="color:${modelColor(mr.model)};font-weight:600">${esc(mr.model)}</span> <span class="muted">not in the vocabulary (OOV)</span></div>`).join('')}
          ${w.overlap != null ? `<div class="muted" style="margin-top:4px">neighbours shared by all ${w.models.length} models: ${w.overlap} / ${r.n}</div>` : ''}</div>`;
      }).join('') + '</div>';
      box.innerHTML = h;
      researchQuestions();
    } catch (e) { err(box, e); }
  }

  async function pairs() {
    const box = $('#ep-out');
    try {
      const r = await api('/api/reuters/pairs?' + new URLSearchParams({ pairs: $('#ep-pairs').value, model: $('#eq-model').value }));
      box.innerHTML = `<div class="tscroll"><table class="grid num"><tr><th>a</th><th>b</th><th>cosine (${esc(r.model)})</th><th>same document (train)</th><th>PMI</th><th></th></tr>
        ${r.pairs.map(p => `<tr><td>${esc(p.a)}</td><td>${esc(p.b)}</td><td>${p.in_model ? f(p.similarity, 3) : '<span class="muted">OOV</span>'}</td><td>${p.in_model ? fmt(p.both) : '–'}</td><td>${p.in_model && p.pmi != null ? f(p.pmi, 1) : '–'}</td><td style="text-align:left">${p.in_model ? `<span class="pbar" style="width:${Math.max(2, Math.round(Math.max(0, p.similarity) * 160))}px"></span>` : ''}</td></tr>`).join('')}</table></div>
        <p class="hint" style="margin-top:6px">rise / fall、buy / sell、profit / loss 都是反義卻 cosine 很高：它們在相同句型與相同新聞中互換出現，向量只知道「用法相同」，不知道「意思相反」。</p>`;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 4. search
   * ------------------------------------------------------------------ */
  function fillMethods() {
    const ms = st.info.methods;
    $('#eq-method').innerHTML = ms.map(m => `<option value="${m.method}"${m.method === st.info.default_method ? ' selected' : ''}>${esc(m.letter)} ${esc(m.method)} — ${esc(m.name)}</option>`).join('');
    const chips = (id, on) => {
      const box = $('#' + id);
      box.innerHTML = ms.map(m => `<label class="${on.includes(m.method) ? 'on' : ''}" title="${esc(m.formula)}"><input type="checkbox" value="${m.method}"${on.includes(m.method) ? ' checked' : ''}><i class="dot" style="background:${colorOf(m.method)};width:10px;height:10px;border-radius:50%;display:inline-block"></i>${esc(m.letter)} ${esc(m.method)} <small class="muted">${esc(m.name)}</small></label>`).join('');
      $$('input', box).forEach(i => i.addEventListener('change', () => i.parentElement.classList.toggle('on', i.checked)));
    };
    chips('ec-methods', ['tfidf', 'w2v-avg', 'w2v-tfidf', 'hybrid']);
    chips('ee-methods', ms.map(m => m.method));
  }

  async function search() {
    const box = $('#eq-out');
    const q = $('#eq').value.trim();
    if (!q) return;
    box.innerHTML = '<p class="muted">searching…</p>';
    try {
      const r = await api('/api/reuters/search?' + new URLSearchParams({ q, method: $('#eq-method').value, model: $('#eq-model').value, alpha: $('#eq-alpha').value, k: $('#eq-k').value }));
      st.lastSearch = r;
      const m = r.measures;
      let h = `<div class="analysis">${r.terms.map(t => t.term ? `<span class="chip ${t.in_model === false && r.method !== 'tfidf' && r.method !== 'tfidf-stem' ? 'stopword' : 'term'}" title="${t.in_model === false ? 'not in the embedding vocabulary (OOV)' : ''}">${esc(t.word)} → <b>${esc(t.term)}</b> <small>df ${fmt(t.df)} · idf ${f(t.idf, 2)}${t.in_model === false ? ' · OOV' : t.count ? ` · trained on ${fmt(t.count)}` : ''}</small></span>` : `<span class="chip stopword">${esc(t.word)}</span>`).join('')}
        ${r.qrows && r.qrows.length ? `<span class="muted">query word weights (${esc(r.method)}): ${r.qrows.filter(x => x.in_model).map(x => `${esc(x.word)} ${f(x.weight, 2)}`).join(', ')}</span>` : ''}</div>
        <div class="summary"><b>${esc(r.letter)} ${esc(r.method_name)}</b>${r.model ? ` · model <b>${esc(r.model)}</b>` : ''}${r.alpha != null ? ` · α = ${r.alpha}` : ''} · ${fmt(r.total)} of ${fmt(st.info.test.documents)} documents scored in ${r.time_ms} ms
        ${r.empty ? '<br><b>no query word is in the embedding vocabulary → empty ranking</b>' : ''}
        ${m ? `<br>evaluation query <b>${esc(r.query_id)}</b> (topic ${tchip(r.topic)}): ${fmt(m.relevant)} relevant · P@5 ${f(m['P@5'], 2)} · P@10 ${f(m['P@10'], 2)} · RR ${f(m.RR, 2)} · nDCG@10 ${f(m['nDCG@10'], 3)} · R@10 ${f(m['R@10'], 3)} · AP ${f(m.AP, 3)}` : ''}</div>`;
      const top = r.results[0] ? r.results[0].score : 1;
      h += r.results.map(d => `<div class="vsm-res emb-res">
          <div class="rank">#${d.rank} · cos <b>${f(d.score, 4)}</b> ${gradeTag(d.grade)} ${topicChips(d.topics)} <span class="muted">${esc(d.doc_id)} · ${d.length} terms · ${esc((d.places || []).join(', '))}</span></div>
          <div class="scorebar"><i style="width:${Math.round(Math.max(0, d.score) / Math.max(top, 1e-9) * 100)}%"></i></div>
          <h3>${esc(d.title)}</h3>
          <div>${r.terms.filter(t => t.term).map(t => `<span class="qw ${d.query_words_in_doc.includes(t.term) ? 'in' : 'out'}" title="${d.query_words_in_doc.includes(t.term) ? 'query word occurs in the document' : 'query word absent from the document'}">${esc(t.term)}</span>`).join('')}
            <span class="muted" style="font-size:12px">lexical overlap ${pct(d.lexical_overlap)}${d.lexical_overlap === 0 ? ' <span class="tag sem">semantic match only</span>' : d.lexical_overlap === 1 ? ' <span class="tag lex">all query words present</span>' : ''}</span></div>
          ${d.explain ? `<table class="explain">${d.explain.filter(x => x.in_model).map(x => `<tr><td><b>${esc(x.word)}</b> ${x.in_doc ? '<span class="muted">in doc</span>' : '<span class="muted">absent</span>'}</td><td>nearest words in this document: ${x.nearest.map(n => `<span title="weight ${n.weight}">${esc(n.word)} <small class="muted">${f(n.similarity, 2)}</small></span>`).join(', ') || '–'}</td></tr>`).join('')}</table>` : ''}
          <details><summary>text</summary><div class="abs">${d.body.map(p => `<p>${esc(p)}</p>`).join('') || '<p class="muted">(BRIEF story without body)</p>'}</div></details>
        </div>`).join('');
      box.innerHTML = h;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 5. compare
   * ------------------------------------------------------------------ */
  async function compare() {
    const box = $('#ec-out');
    const q = $('#ec-q').value.trim();
    const methods = picked('ec-methods');
    if (!q || !methods.length) return;
    box.innerHTML = '<p class="muted">comparing…</p>';
    try {
      const r = await api('/api/reuters/compare?' + new URLSearchParams({ q, methods: methods.join(','), model: $('#eq-model').value, alpha: $('#eq-alpha').value, k: $('#ec-k').value }));
      let h = `<div class="summary">query terms: ${r.query_terms.map(esc).join(', ')}${r.model ? ` · model ${esc(r.model)}` : ''} · base = <b>${esc(methodLabel(r.base))}</b>${r.query_id ? ` · evaluation query ${esc(r.query_id)}` : ''}</div>
        <div class="tscroll"><table class="grid num"><tr><th>method</th><th>name</th><th>scored docs</th><th>overlap@k</th><th>Kendall τ</th><th>詞彙重疊 (top-k)</th><th>無查詢詞</th>${r.query_id ? '<th>P@10</th><th>RR</th><th>nDCG@10</th>' : ''}</tr>
        ${r.methods.map(m => `<tr><td><b style="color:${colorOf(m.method)}">${esc(m.letter)} ${esc(m.method)}</b></td><td>${esc(m.name)}</td><td>${fmt(m.total)}</td><td>${f(m.overlap, 2)}</td><td>${f(m.tau, 2)}</td><td>${pct(m.mean_lexical_overlap)}</td><td>${m.no_query_word}</td>${m.measures ? `<td>${f(m.measures['P@10'], 2)}</td><td>${f(m.measures.RR, 2)}</td><td>${f(m.measures['nDCG@10'], 3)}</td>` : ''}</tr>`).join('')}</table></div>
        <div class="vcols">${r.methods.map(m => `<div class="vcol"><h4><span style="color:${colorOf(m.method)}">${esc(m.letter)} ${esc(m.method)}</span></h4><ol>
          ${m.top.map(t => `<li class="${t.grade ? 'rel' + t.grade : ''}" title="${esc(t.title)}"><span class="mv">${f(t.score, 3)}</span> ${esc(t.title.length > 46 ? t.title.slice(0, 46) + '…' : t.title)}<br><span class="mv">${(t.topics || []).map(tchip).join('') || '<span class="muted">no topic</span>'} ${t.query_words.length ? t.query_words.map(esc).join(', ') : '<span class="tag sem">no query word</span>'} · base rank ${t.rank_base || '–'}</span></li>`).join('')}</ol></div>`).join('')}</div>
        <p class="hint" style="margin-top:6px">綠底 = 相關（深綠 grade 2）。TF-IDF 的前 k 名一定含查詢詞；嵌入法可以把不含任何查詢詞的新聞排到前面——若那些新聞是相關的，就是語意匹配的收益；若不是，就是「情境相近但主題不同」的代價。</p>`;
      box.innerHTML = h;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 6. evaluation
   * ------------------------------------------------------------------ */
  async function loadQueries() {
    const box = $('#ee-queries');
    try {
      const r = await api('/api/reuters/queries');
      st.queries = r;
      $('#eq-examples').innerHTML = '<span class="muted">evaluation queries:</span> ' + r.queries.map(q => `<a href="#" data-q="${esc(q.text)}" title="${esc(q.kind_name)} · ${esc(q.description)} · ${q.relevant} relevant">${esc(q.id)}</a>`).join('');
      $$('#eq-examples a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#eq').value = a.dataset.q; $('#ec-q').value = a.dataset.q; search(); compare(); }));
      box.innerHTML = `<div class="tscroll"><table class="grid qtab"><tr><th>id</th><th>query</th><th>topic</th><th>kind</th><th style="text-align:right">relevant</th><th style="text-align:right">grade 2</th><th>why this wording</th><th></th></tr>
        ${r.queries.map(q => `<tr><td>${esc(q.id)}</td><td class="t"><b>${esc(q.text)}</b></td><td>${tchip(q.topic)}</td><td>${esc(q.kind_name)}</td><td style="text-align:right">${fmt(q.relevant)}</td><td style="text-align:right">${fmt(q.grade2)}</td><td class="t" style="font-size:12px">${esc(q.description)}</td><td><a href="#" data-rel="${esc(q.id)}">relevant set</a> · <a href="#" data-run="${esc(q.text)}">search</a></td></tr>`).join('')}</table></div>
        <p class="hint" style="margin-top:6px">${fmt(r.documents)} test documents; file <code>${esc(r.file)}</code>. ${esc(r.notes)}</p>`;
      $$('a[data-rel]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); relevantSet(a.dataset.rel); }));
      $$('a[data-run]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#eq').value = a.dataset.run; $('#ec-q').value = a.dataset.run; search(); compare(); $('#emb-s4').scrollIntoView(); }));
    } catch (e) { err(box, e); }
  }

  async function relevantSet(qid) {
    const box = $('#ee-rel');
    box.innerHTML = '<p class="muted">loading…</p>';
    try {
      const method = $('#eq-method').value;
      const r = await api('/api/reuters/relevant?' + new URLSearchParams({ id: qid, method, model: $('#eq-model').value, alpha: $('#eq-alpha').value }));
      const m = r.measures;
      box.innerHTML = `<h4 style="margin:12px 0 4px">Relevant set of ${esc(r.query.id)} — “${esc(r.query.text)}” (${fmt(r.total_relevant)} documents, first ${r.relevant.length} by rank under <b>${esc(methodLabel(method))}</b>: P@10 ${f(m['P@10'], 2)}, RR ${f(m.RR, 2)}, nDCG@10 ${f(m['nDCG@10'], 3)}; ${fmt(r.retrieved)} documents scored)</h4>
        <div class="tscroll"><table class="grid"><tr><th style="text-align:right">rank</th><th>grade</th><th>document</th><th>topics</th></tr>
        ${r.relevant.map(d => `<tr><td style="text-align:right">${d.rank || '<span class="muted">not scored</span>'}</td><td>${gradeTag(d.grade)}</td><td class="t">${esc(d.title)} <span class="muted">${esc(d.doc_id)}</span></td><td style="text-align:left">${topicChips(d.topics)}</td></tr>`).join('')}</table></div>`;
      box.scrollIntoView({ block: 'start' });
    } catch (e) { err(box, e); }
  }

  async function evaluate() {
    const box = $('#ee-out');
    const methods = picked('ee-methods'), models = picked('ee-models');
    if (!methods.length) return;
    box.innerHTML = '<p class="muted">evaluating…</p>';
    try {
      const r = await api('/api/reuters/eval?' + new URLSearchParams({ methods: methods.join(','), models: models.join(','), k: 10, alpha: $('#ee-alpha').value }));
      st.evalData = r;
      const best = {};
      MEASURES.forEach(k => { best[k] = Math.max(...r.runs.map(s => s.mean[k])); });
      const runLabel = s => `<b style="color:${colorOf(s.method)}">${esc(s.letter)} ${esc(s.method)}</b>${s.model ? ` <span style="color:${modelColor(s.model)}">· ${esc(s.model)}</span>` : ''}${s.alpha != null ? ` <small class="muted">α ${s.alpha}</small>` : ''}`;
      let h = `<div class="tscroll"><table class="grid num"><tr><th>run</th>${MEASURES.map(k => `<th>${k}</th>`).join('')}</tr>
        ${r.runs.map(s => `<tr><td>${runLabel(s)}</td>${MEASURES.map(k => `<td class="${s.mean[k] === best[k] ? 'best' : ''}">${f(s.mean[k], 3)}</td>`).join('')}</tr>`).join('')}</table></div>
        <p class="hint">mean over ${r.queries.length} queries; best value per column in bold. MAP 與 R@10 對 TF-IDF 不利，因為它只對含查詢詞的文件給分（c03「petroleum…」只檢回 ${fmt((st.lastSearch && st.lastSearch.query_id === 'c03' && st.lastSearch.method === 'tfidf') ? st.lastSearch.total : 284)} 篇）；嵌入法對每篇文件都有分數。</p>
        <div class="row" style="margin:8px 0"><label>Measure <select id="ee-measure">${MEASURES.map(k => `<option${k === 'P@10' ? ' selected' : ''}>${k}</option>`).join('')}</select></label></div>
        <div class="two"><div id="ee-chart"></div><div id="ee-kind"></div></div>
        <div class="two" style="margin-top:10px"><div id="ee-topic"></div><div id="ee-alpha-plot"></div></div>
        <div id="ee-perq"></div>`;
      box.innerHTML = h;
      const draw = () => {
        const k = $('#ee-measure').value;
        $('#ee-chart').innerHTML = `<h4>${esc(k)} per run</h4>` + hbarChart(r.runs.map(s => [`${s.letter} ${s.model ? s.model.replace(/-w5-m5/, '') : s.method}`, +s.mean[k].toFixed(3)]), { width: 520, barH: 18, labelW: 150 });
        $('#ee-kind').innerHTML = `<h4>${esc(k)} by query kind（用字離文件詞彙越遠，往右）</h4><table class="grid num"><tr><th>run</th>${r.kinds.map(x => `<th>${esc(x.name)}</th>`).join('')}</tr>
          ${r.runs.map(s => `<tr><td>${runLabel(s)}</td>${r.kinds.map(x => { const vals = r.runs.map(y => y.by_kind[x.kind] ? y.by_kind[x.kind][k] : -1); const v = s.by_kind[x.kind] ? s.by_kind[x.kind][k] : null; return `<td class="${v != null && v === Math.max(...vals) ? 'best' : ''}">${f(v, 3)}</td>`; }).join('')}</tr>`).join('')}</table>`;
        $('#ee-topic').innerHTML = `<h4>${esc(k)} by topic</h4><table class="grid num"><tr><th>run</th>${r.topics.map(t => `<th>${tchip(t.topic)}</th>`).join('')}</tr>
          ${r.runs.map(s => `<tr><td>${runLabel(s)}</td>${r.topics.map(t => { const vals = r.runs.map(y => y.by_topic[t.topic] ? y.by_topic[t.topic][k] : -1); const v = s.by_topic[t.topic] ? s.by_topic[t.topic][k] : null; return `<td class="${v != null && v === Math.max(...vals) ? 'best' : ''}">${f(v, 3)}</td>`; }).join('')}</tr>`).join('')}</table>`;
        $('#ee-perq').innerHTML = `<h4>${esc(k)} per query</h4><div class="tscroll"><table class="grid num"><tr><th>query</th><th>kind</th>${r.runs.map(s => `<th style="color:${colorOf(s.method)}" title="${esc(s.label)}">${esc(s.letter)}${s.model ? '<br><small>' + esc(s.model.replace(/-w5-m5/, '')) + '</small>' : ''}</th>`).join('')}<th>relevant</th></tr>
          ${r.queries.map((q, i) => { const vals = r.runs.map(s => s.per_query[i][k]); const mx = Math.max(...vals);
            return `<tr><td title="${esc(q.text)}">${esc(q.id)} ${tchip(q.topic)} <span class="muted">${esc(q.text.length > 40 ? q.text.slice(0, 40) + '…' : q.text)}</span></td><td>${esc(q.kind_name)}</td>${vals.map(v => `<td class="${v === mx ? 'best' : ''}">${f(v, 3)}</td>`).join('')}<td>${fmt(q.relevant)}</td></tr>`; }).join('')}</table></div>`;
      };
      $('#ee-measure').addEventListener('change', draw);
      draw();
      alphaSweep();
      researchQuestions();
    } catch (e) { err(box, e); }
  }

  async function alphaSweep() {
    const box = $('#ee-alpha-plot');
    if (!box) return;
    try {
      const r = await api('/api/reuters/alpha?' + new URLSearchParams({ model: $('#eq-model').value }));
      st.alpha = r;
      box.innerHTML = `<h4>方法 D：α·TF-IDF + (1 − α)·B 的 α 掃描（${esc(r.model)}）— 最佳 nDCG@10 在 α = ${r.best_alpha}</h4><div id="ee-alpha-svg"></div>`;
      plot($('#ee-alpha-svg'), { height: 240, width: 560, xlabel: 'α  (0 = weighted embedding only, 1 = TF-IDF only)', ylabel: 'mean over queries', yzero: true, xzero: true,
        series: [{ name: 'nDCG@10', color: COLORS.hybrid, pts: r.rows.map(x => [x.alpha, x['nDCG@10']]) }, { name: 'P@10', color: COLORS.tfidf, pts: r.rows.map(x => [x.alpha, x['P@10']]) }, { name: 'MRR', color: COLORS['w2v-tfidf'], pts: r.rows.map(x => [x.alpha, x.MRR]) }],
        tipX: v => `α = ${v}`, tipY: v => f(v, 3) });
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 7. research questions from the live numbers
   * ------------------------------------------------------------------ */
  function researchQuestions() {
    const box = $('#er-out');
    const r = st.evalData, nb = st.neighbours;
    if (!r) return;
    const run = (m, model) => r.runs.find(s => s.method === m && (!model || s.model === model));
    const runsOf = m => r.runs.filter(s => s.method === m);
    const A = run('tfidf'), A0 = run('tfidf-stem');
    const Bs = runsOf('w2v-tfidf'), Cs = runsOf('w2v-avg'), Ds = runsOf('hybrid');
    const B = Bs[0], C = Cs.find(c => B && c.model === B.model) || Cs[0], D = Ds.find(d => B && d.model === B.model) || Ds[0];
    const d = (a, b, k) => (a && b) ? `${k} ${f(a.mean[k], 3)} → ${f(b.mean[k], 3)} (${b.mean[k] >= a.mean[k] ? '+' : ''}${((b.mean[k] - a.mean[k]) / (a.mean[k] || 1) * 100).toFixed(0)}%)` : '–';
    const kind = (s, kd, k) => (s && s.by_kind[kd] ? f(s.by_kind[kd][k], 2) : '–');
    const sg = Bs.find(s => /^sg-/.test(s.model || '')), cbow = Bs.find(s => /^cbow-/.test(s.model || ''));
    const wins = (a, b, k) => (a && b) ? a.per_query.filter((q, i) => q[k] > b.per_query[i][k]).length : 0;
    const ties = (a, b, k) => (a && b) ? a.per_query.filter((q, i) => q[k] === b.per_query[i][k]).length : 0;
    let nbText = '';
    if (nb) {
      const w = word => nb.words.find(x => x.word === word);
      const first = (word, i = 0) => { const x = w(word); const m = x && x.models.find(y => y.in_model); return m && m.neighbours[i] ? `${m.neighbours[i].word} (${f(m.neighbours[i].similarity, 2)})` : '–'; };
      const list = word => { const x = w(word); const m = x && x.models.find(y => y.in_model); return m ? m.neighbours.map(n => n.word).join(', ') : '–'; };
      nbText = `<p><b>RQ1 Word-level：哪些詞在向量空間中接近，為什麼。</b> 以第 3 節的結果（第一個模型）：<span class="n">oil</span> → ${esc(list('oil'))}；<span class="n">profit</span> → ${esc(list('profit'))}；<span class="n">petroleum</span> → ${esc(list('petroleum'))}；<span class="n">grain</span> → ${esc(list('grain'))}。
        合理的語意／用法關係（oil–crude、profit–loss–pretax、grain–wheat–corn）來自相同句型；<b>情境共現</b>標記的鄰近詞（同篇多、PMI 高）是因為常在同一類新聞裡出現，不是同義；
        <b>不符預期</b>的例子是 petroleum：在路透語料裡它大多是公司名的一部分（… Petroleum Corp、Pacific Petroleum），所以鄰近詞是公司名／礦業詞而非 oil、crude——向量學到的是「這個詞的上下文」，而語料裡 petroleum 的上下文是併購新聞。profit / loss 反義卻相近，說明向量相近 ≠ 同義。</p>`;
    }
    box.innerHTML = `<div class="rq">${nbText}
      <p><b>RQ2 詞彙匹配 vs 語意匹配（document-level）。</b> TF-IDF 只對含查詢詞的文件給分：在 ${r.queries.length} 個查詢中，依查詢用字分四類，P@10 為 主題詞 ${kind(A, 'topic', 'P@10')} / 路透慣用詞 ${kind(A, 'domain', 'P@10')} / 同義改寫 ${kind(A, 'synonym', 'P@10')} / 情境描述 ${kind(A, 'semantic', 'P@10')}（A）；
        加權詞向量（B${B ? '，' + esc(B.model) : ''}）為 ${kind(B, 'topic', 'P@10')} / ${kind(B, 'domain', 'P@10')} / ${kind(B, 'synonym', 'P@10')} / ${kind(B, 'semantic', 'P@10')}。
        查詢用字越接近文件詞彙，TF-IDF 越難被超越；用字一旦離開文件詞彙（petroleum、cereal、energy exploration），TF-IDF 的 P@10 掉得比嵌入法多。第 4 節的 c03「petroleum production and output」是教科書例子：TF-IDF 只找得到含 petroleum / production / output 的新聞，其中多半是工業產出（ipi）新聞；嵌入法把不含這些詞的 crude 新聞排上來。</p>
      <p><b>RQ3 核心比較：A TF-IDF vs B TF-IDF 加權詞向量。</b> ${d(A, B, 'P@10')}、${d(A, B, 'MRR')}、${d(A, B, 'nDCG@10')}、${d(A, B, 'MAP')}。逐題（P@10）：B 勝 ${wins(B, A, 'P@10')} 題、A 勝 ${wins(A, B, 'P@10')} 題、平手 ${ties(A, B, 'P@10')} 題。
        依主題：${r.topics.map(t => `${t.topic} ${A && A.by_topic[t.topic] ? f(A.by_topic[t.topic]['nDCG@10'], 2) : '–'} → ${B && B.by_topic[t.topic] ? f(B.by_topic[t.topic]['nDCG@10'], 2) : '–'}`).join('；')}（nDCG@10）。
        結論必須分情況：B 在同義／情境查詢上較好，在主題詞查詢上不一定；MAP 的大差距主要反映 TF-IDF 的召回上限（未含查詢詞的相關文件永遠排不進來），而非前十名的品質。${A0 ? ` 參考：Porter stemming 的 TF-IDF（A₀）P@10 ${f(A0.mean['P@10'], 3)}、nDCG@10 ${f(A0.mean['nDCG@10'], 3)}——stemming 把 prices / price、exports / export 合併，是 TF-IDF 自己的「有限語意匹配」。` : ''}</p>
      <p><b>RQ4 CBOW vs skip-gram。</b> ${sg && cbow ? `同樣的 B 方法：skip-gram（${esc(sg.model)}）P@10 ${f(sg.mean['P@10'], 3)}、nDCG@10 ${f(sg.mean['nDCG@10'], 3)}、MRR ${f(sg.mean.MRR, 3)}；CBOW（${esc(cbow.model)}）P@10 ${f(cbow.mean['P@10'], 3)}、nDCG@10 ${f(cbow.mean['nDCG@10'], 3)}、MRR ${f(cbow.mean.MRR, 3)}。逐題 P@10：sg 勝 ${wins(sg, cbow, 'P@10')}、CBOW 勝 ${wins(cbow, sg, 'P@10')}、平手 ${ties(sg, cbow, 'P@10')}。` : '在第 6 節同時勾選一個 skip-gram 與一個 CBOW 模型即可比較。'}
        在 763K 詞元的小語料上兩者差距小，且會隨 seed、維度與 epochs 變動（第 2 節可訓練更多變體）；不能預設 skip-gram 一定較好。第 3 節顯示兩種架構的 Top-5 鄰近詞重疊通常只有 2–3 個，但文件層級的排名差異比詞層級小——文件向量是上百個詞向量的加權和，個別鄰近詞的差異被平均掉了。</p>
      <p><b>RQ5 加權方式與融合。</b> 詞向量平均（C）vs TF-IDF 加權平均（B）：${d(C, B, 'P@10')}、${d(C, B, 'nDCG@10')}、${d(C, B, 'MRR')}——不加權時 said、mln、dlrs 這類高頻詞拉著每篇文件的向量往同一個方向，查詢向量跟所有文件都「差不多像」；idf 加權後文件向量跟著有辨識力的詞走。
        融合（D，α = ${D ? D.alpha : '–'}）：${d(B, D, 'P@10')}、${d(B, D, 'nDCG@10')}、${d(B, D, 'MRR')}${st.alpha ? `；α 掃描的最佳 nDCG@10 在 α = ${st.alpha.best_alpha}（純 TF-IDF ${f(st.alpha.rows[st.alpha.rows.length - 1]['nDCG@10'], 3)}、純 B ${f(st.alpha.rows[0]['nDCG@10'], 3)}）` : ''}。
        限制：加權平均不保留詞序、否定與片語（"not acquire" 與 "acquire" 向量幾乎相同）；相關性來自 TOPICS 標籤，一篇講石油公司併購的新聞對 crude 查詢可能只標 acq；結果只能推論到小型新聞語料，不能直接推到大型現代語言模型。</p>
    </div>`;
  }

  /* ------------------------------------------------------------------ *
   * wiring
   * ------------------------------------------------------------------ */
  $('#et-go').addEventListener('click', train);
  $('#en-go').addEventListener('click', neighbours);
  $('#en-words').addEventListener('keydown', e => { if (e.key === 'Enter') neighbours(); });
  $('#ep-go').addEventListener('click', pairs);
  $('#eq-go').addEventListener('click', search);
  $('#eq').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
  $('#eq-method').addEventListener('change', () => { if (st.lastSearch) search(); });
  $('#eq-model').addEventListener('change', () => { if (st.lastSearch) search(); });
  $('#ec-go').addEventListener('click', compare);
  $('#ec-q').addEventListener('keydown', e => { if (e.key === 'Enter') compare(); });
  $$('#emb-s5 button[data-exp]').forEach(b => b.addEventListener('click', () => { setPicked('ec-methods', b.dataset.exp.split(',')); compare(); }));
  $('#ee-go').addEventListener('click', evaluate);

  async function init() {
    if (st.loaded) return;
    st.loaded = true;
    await loadInfo();
    if (!st.info) return;
    fillMethods();
    renderModels(st.info.models, st.info.active);
    await loadQueries();
    if (!st.info.models.length) { refreshModels(); return; }
    await search();
    await Promise.all([neighbours(), pairs(), compare()]);
    await evaluate();
    refreshModels();
  }
  document.addEventListener('tab', e => { if (e.detail === 'emb') init(); });
  if (location.hash === '#emb' || location.hash.startsWith('#emb-')) { showTab('emb'); }
})();
