/* Project #4 walkthrough (Word Embedding tab): the tab is a six-step stepper (data, models, word level, document
   level, evaluation, conclusions) that shows one step at a time; cards carry data-step, long explanations are folded
   behind a button and the full tools are collapsed "advanced" cards.  Steps 3-5 are the showcase (#emb-s0 / #emb-s0b):
   one-click special queries that answer the points of the spec (A vs B duel with a relevance strip per method, query pairs for the limits of averaging,
   word-level neighbours) and an at-a-glance view of the experiment data (headline deltas, P@10 by query wording,
   per-query dumbbell, model variants).  Every number is fetched live from /api/reuters/*; only the "what to look
   at" sentences are written by hand. */
(() => {
  const { $, $$, esc, fmt, api, showTab } = window.IR;
  const f = (v, d = 2) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const f0 = v => f(v, 2).replace(/^0\./, '.');
  const METHODS = [
    { id: 'tfidf', letter: 'A', name: 'TF-IDF', v: 'var(--m-a)' },
    { id: 'w2v-tfidf', letter: 'B', name: 'TF-IDF 加權詞向量', v: 'var(--m-b)' },
    { id: 'w2v-avg', letter: 'C', name: '詞向量平均', v: 'var(--m-c)' },
    { id: 'hybrid', letter: 'D', name: '融合 α·A + (1−α)·B', v: 'var(--m-d)' },
  ];
  const M = Object.fromEntries(METHODS.map(m => [m.id, m]));
  const KIND_ZH = { topic: '主題詞', domain: '路透慣用詞', synonym: '同義／改寫', semantic: '情境描述' };
  const TOPICS = ['earn', 'acq', 'crude', 'grain', 'trade'];
  const tchip = t => `<span class="tchip ${esc(t)}">${esc(t)}</span>`;
  const key = m => `<i class="sc-key" style="background:${M[m].v}"></i>`;
  const st = { models: [], sg: '', cbow: '', evalData: null, alpha: null, queries: [], ready: null, measure: 'P@10', sort: 'kind', view: 'kind', step: 0, last: {}, titles: false, show: { 'w2v-avg': false, hybrid: false } };
  const shown = id => id === 'tfidf' || id === 'w2v-tfidf' || st.show[id];
  /* the two optional comparisons (not in the spec), each with its own checkbox */
  const extraPicks = () => `<span class="sc-extra"><span class="muted">自行加入的對照：</span>${[['w2v-avg', 'C 詞向量平均'], ['hybrid', 'D 融合']].map(([id, t]) => `<label class="check"><input type="checkbox" data-extra="${id}"${st.show[id] ? ' checked' : ''}> ${key(id)}${t}</label>`).join('')}</span>`;

  /* ---------------- the six steps ---------------- */
  const STEPS = [
    { n: 1, name: '資料', ask: '用哪些新聞？怎麼知道一篇新聞「相關」？' },
    { n: 2, name: '詞向量', ask: '在 train 新聞上訓練 word2vec：skip-gram 與 CBOW。' },
    { n: 3, name: 'Word-level', ask: '哪些詞的向量接近？接近的原因合理嗎？', h: '選一組詞，看它的最近鄰' },
    { n: 4, name: 'Document-level', ask: '同一個查詢，A（TF-IDF）與 B（詞向量）誰把相關新聞排在前面？', h: '選一個查詢，看 A / B 對決' },
    { n: 5, name: '評估', ask: '20 個查詢平均起來差多少？差在哪一類查詢？' },
    { n: 6, name: '結論', ask: '五個研究問題的答案。' },
  ];

  /* ---------------- the scenarios ---------------- */
  const GROUPS = [
    { title: '', items: [
      { id: 'c03', q: 'petroleum production and output', label: 'petroleum production and output', tag: 'B 贏：語意匹配', star: true,
        note: '新聞寫 crude / oil，不寫 petroleum：A 找到工業產出新聞，B 找到石油產量新聞，連<b>不含任何查詢詞</b>的都排得進來。' },
      { id: 'c04', q: 'energy exploration and drilling', label: 'energy exploration and drilling', tag: 'B 贏：公司名的干擾',
        note: 'A 的前 10 名多是鑽井公司的<b>財報</b>：公司名裡有 drilling / energy 就命中。' },
      { id: 't03', q: 'import tariffs and export restrictions', label: 'import tariffs and export restrictions', tag: 'A 贏：B 帶進鄰近主題',
        note: 'B 的前 10 名混進柳橙汁、油籽、植物油的<b>貿易爭端</b>新聞：情境相同，但編目員標的主題是商品，不是 trade。語意匹配會把鄰近主題一起帶進來。' },
      { id: 'anto', pair: ['profit', 'loss'], topic: 'earn', mark: 'LOSS', open: true, label: 'profit ↔ loss', tag: '限制：反義詞',
        note: '反義詞的向量很近：用 B 查 profit，會排出標題寫著 <b>LOSS</b> 的財報；而 P@10 全是 1.00，指標看不出來。' },
      { id: 'oov', q: 'Cajamarquilla', label: 'Cajamarquilla', tag: '限制：沒有向量的詞',
        note: '低於 min_count 的詞沒有向量：B 回傳<b>空排名</b>，A 字面出現一次就找得到。' },
    ] },
  ];
  const WORDS = [
    { id: 'w-oil', words: 'oil crude', label: 'oil ／ crude', tag: '合理的語意關係',
      note: '同一個語意場的詞在相同句型中互換出現（"X prices"、"X output rose"），向量自然靠近。' },
    { id: 'w-opec', words: 'opec', label: 'opec', tag: '鄰居是同類新聞的常客',
      note: '看 opec 的鄰居：人名（lukman、nazer）、國名、產量用語，沒有一個能替換 opec，卻都常和它寫在同一篇新聞裡。對照 oil 的鄰居 crude：可以替換。' },
    { id: 'w-pet', words: 'petroleum', label: 'petroleum', tag: '不符預期', star: true,
      note: '預期 oil、crude，實際是公司名：petroleum 在這份語料裡多半是<b>公司名的一部分</b>。' },
    { id: 'w-rare', words: 'cereal tariff', label: 'cereal ／ tariff', tag: '稀有詞',
      note: '只訓練過幾十次：鄰居看不出關係、cosine 卻特別高，兩個模型的結果也不一致。' },
    { id: 'w-anto', pairs: 'profit,loss; rise,fall; buy,sell; deficit,surplus; oil,petroleum; grain,cereal', label: '反義詞對的 cosine', tag: '相近 ≠ 同義',
      note: '反義詞對的 cosine 比真正的同義詞（oil / petroleum）還高：向量只知道「用法相同」。' },
    { id: 'w-pmi', pmi: 'crude,barrel; crude,crudes; opec,bpd; opec,lukman; opec,coffee', label: 'crudes vs barrel', tag: 'PMI 高 ≠ 證據多',
      note: 'PMI 量的是「一起出現的<b>比例</b>比隨機高幾倍」，不是篇數：只出現 12 篇的 crudes，PMI 反而比同篇 71 篇的 barrel 高。要和「同篇」一起看。' },
  ];
  const ALL = Object.fromEntries([...GROUPS.flatMap(g => g.items), ...WORDS].map(s => [s.id, s]));
  const stepOf = s => (s.words || s.pairs || s.pmi ? 3 : 4);

  /* ---------------- shared hover tooltip ---------------- */
  const tip = document.createElement('div');
  tip.className = 'sc-tip hidden';
  document.body.appendChild(tip);
  document.addEventListener('mousemove', e => {
    const el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (!el || !el.closest('#emb-s0, #emb-s0b')) { tip.classList.add('hidden'); return; }
    tip.innerHTML = el.dataset.tip;
    tip.classList.remove('hidden');
    const z = parseFloat(document.body.style.zoom) || 1;       // body zoom (demo.js) scales fixed children
    const x = e.clientX / z, y = e.clientY / z, w = tip.offsetWidth, vw = window.innerWidth / z;
    tip.style.left = Math.max(6, Math.min(x + 14, vw - w - 10)) + 'px';
    tip.style.top = (y + 16) + 'px';
  });

  /* ---------------- A/B duel ---------------- */
  const cell = (t, nq) => {
    const g = t.grade, lex = t.query_words.length;
    const cls = g == null ? 'u' : 'g' + g;
    const rel = g == null ? '' : g === 2 ? '相關（唯一主題）' : g === 1 ? '相關（多主題之一）' : '不相關';
    return `<span class="sc-cell ${cls}${lex ? '' : ' nolex'}" data-tip="<b>#${t.rank}</b> ${esc(esc(t.title))}<br>${esc((t.topics || []).join(', ') || 'no topic')}${rel ? ' · ' + rel : ''}<br>含查詢詞 ${lex} / ${nq}${lex ? '：' + esc(t.query_words.join(', ')) : '（純語意匹配）'}">${t.rank}</span>`;
  };
  const strip = (m, nq) => (m.top.length ? m.top.map(t => cell(t, nq)).join('') : '<span class="muted">（空排名：查詢詞都沒有詞向量）</span>');
  const other = (t, m, o) => {
    const r = t.ranks[o];
    return r === 0 ? `<span class="sc-rk out">${M[o].letter} 未檢回</span>` : r > 10 ? `<span class="sc-rk far">${M[o].letter} 排 #${r}</span>` : `<span class="sc-rk">${M[o].letter} #${r}</span>`;
  };
  const title = (t, mark) => { const x = esc(t.title.replace(/\s+/g, ' ')); return mark ? x.replace(new RegExp('\\b' + mark + '\\b', 'gi'), y => `<mark>${y}</mark>`) : x; };
  const column = (m, o, nq, mark, label) => `<div class="sc-col"><h4>${key(m.method)}${label || M[m.method].letter + ' ' + esc(M[m.method].name)}</h4>
    ${m.top.length ? `<ol>${m.top.map(t => `<li class="${t.grade == null ? '' : 'g' + t.grade}"><span class="sc-t">${title(t, mark)}</span>
      <span class="sc-m">${(t.topics || []).slice(0, 3).map(tchip).join('') || '<span class="muted">no topic</span>'}${t.query_words.length ? `<span class="muted">${esc(t.query_words.join(', '))}</span>` : '<span class="tag sem">無查詢詞</span>'}${o ? other(t, m, o) : ''}</span></li>`).join('')}</ol>`
      : '<p class="muted">沒有任何文件得到分數。</p>'}</div>`;
  const legend = graded => `<div class="sc-legend">${graded ? '<span><i class="sc-cell g2 mini"></i>相關</span><span><i class="sc-cell g1 mini"></i>相關（多主題之一）</span><span><i class="sc-cell g0 mini"></i>不相關</span>' : '<span><i class="sc-cell u mini"></i>未判定相關</span>'}
    <span><i class="sc-cell u nolex mini"></i>虛線框 = 不含任何查詢詞</span><span class="muted">滑鼠移到格子看標題</span></div>`;
  const titlesBox = (inner, s) => `<details class="sc-titles"${st.titles || s.open ? ' open' : ''}><summary>前 10 名的新聞標題</summary>${inner}</details>`;
  const bindTitles = box => { const d = box.querySelector('.sc-titles'); if (d) d.addEventListener('toggle', () => { st.titles = d.open; }); };

  async function compare(q, topic, methods) {
    return api('/api/reuters/compare?' + new URLSearchParams({ q, methods: methods || METHODS.map(m => m.id).join(','), model: $('#sc-model').value, alpha: $('#sc-alpha').value, k: 10, topic: topic || '' }));
  }
  const head = (s, r) => `<div class="sc-head"><div class="sc-q">${esc(s.pair ? s.pair.join('  ↔  ') : r.query)}</div>
    <div class="sc-meta">${r.topic ? `<span class="muted">相關 =</span>${tchip(r.topic)}<span class="muted">新聞（${fmt(r.relevant)} 篇）</span>` : '<span class="muted">未判定相關</span>'}</div></div>
    ${s.note ? `<p class="sc-note">${s.note}</p>` : ''}`;
  const words = r => `<div class="sc-words"><span class="muted">前處理後：</span>${r.query_words.length ? r.query_words.map(w => `<span class="chip ${w.in_model ? 'term' : 'oov'}">${esc(w.word)}${w.in_model ? '' : ' <small>OOV 無向量</small>'}</span>`).join('') : '<span class="muted">（沒有剩下任何詞）</span>'}</div>`;
  /* which embedding model a run used, e.g. "skip-gram d100" (window / epochs only when they differ from 5 / 5) */
  const modelTag = (name, method) => {
    if (method === 'tfidf' || !name) return '';
    const m = st.models.find(x => x.name === name);
    const t = m ? `${m.model === 'cbow' ? 'CBOW' : 'skip-gram'} d${m.dim}${m.window !== 5 ? ' w' + m.window : ''}${m.epochs !== 5 ? ' e' + m.epochs : ''}` : name;
    return ` <small class="sc-model" title="${esc(name)}">· ${esc(t)}</small>`;
  };
  const fail = (box, e) => { box.innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; };

  async function duel(s, box) {
    try {
      const r = await compare(s.q, s.topic);
      const nq = r.query_terms.length, graded = !!r.query_id;
      const A = r.methods.find(m => m.method === 'tfidf'), B = r.methods.find(m => m.method === 'w2v-tfidf');
      let verdict;
      if (graded && A.measures && B.measures) {
        const a = A.measures['P@10'], b = B.measures['P@10'];
        verdict = `<div class="sc-verdict"><span class="big">P@10　${key('tfidf')}A ${f(a)}<em>→</em>${key('w2v-tfidf')}B ${f(b)}</span>
          <span>${b > a ? '▲ B 較好' : a > b ? '▼ A 較好' : '＝ 相同'}${B.no_query_word ? ` · B 的前 10 名有 <b>${B.no_query_word}</b> 篇不含任何查詢詞` : ''}</span></div>`;
      } else {
        verdict = `<div class="sc-verdict"><span>A 找到 <b>${fmt(A.total)}</b> 篇、B 找到 <b>${fmt(B.total)}</b> 篇${B.no_query_word ? ` · B 的前 10 名有 <b>${B.no_query_word}</b> 篇不含任何查詢詞` : ''}</span></div>`;
      }
      box.innerHTML = head(s, r) + (r.query_words.some(w => !w.in_model) ? words(r) : '') + verdict + `
        <table class="sc-strips sc-duel"><tr><th>方法</th><th>前 10 名（左 = 第 1 名）</th><th>P@10</th><th>nDCG@10</th></tr>
        ${r.methods.map(m => `<tr data-m="${m.method}"${shown(m.method) ? '' : ' hidden'}><td class="sc-mname">${key(m.method)}<b>${M[m.method].letter}</b> ${esc(M[m.method].name)}${modelTag(r.model, m.method)}</td><td class="sc-strip">${strip(m, nq)}</td>
          <td>${m.measures ? f(m.measures['P@10']) : '–'}</td><td>${m.measures ? f(m.measures['nDCG@10'], 3) : '–'}</td></tr>`).join('')}</table>
        <div class="sc-cd">${extraPicks()}</div>
        ${legend(graded)}
        ${titlesBox(`<div class="sc-cols">${column(A, 'w2v-tfidf', nq)}${column(B, 'tfidf', nq)}</div>`, s)}`;
      bindTitles(box);
      box.querySelectorAll('.sc-cd input').forEach(i => i.addEventListener('change', () => { st.show[i.dataset.extra] = i.checked; box.querySelector(`.sc-duel tr[data-m="${i.dataset.extra}"]`).hidden = !i.checked; }));
    } catch (e) { fail(box, e); }
  }

  /* ---------------- two queries, same methods ---------------- */
  async function pairDuel(s, box) {
    try {
      const rs = await Promise.all(s.pair.map(q => compare(q, s.topic, 'tfidf,w2v-tfidf')));
      const top = (r, m) => r.methods.find(x => x.method === m).top;
      const shared = m => { const a = new Set(top(rs[0], m).map(t => t.doc_id)); return top(rs[1], m).filter(t => a.has(t.doc_id)).length; };
      const sameTerms = rs[0].query_terms.join() === rs[1].query_terms.join();
      let cos = '';
      if (s.pair.every(q => /^\w+$/.test(q))) {
        const p = await api('/api/reuters/pairs?' + new URLSearchParams({ pairs: s.pair.join(','), model: $('#sc-model').value }));
        if (p.pairs[0] && p.pairs[0].in_model) cos = `兩個詞的向量 cosine <b>${f(p.pairs[0].similarity)}</b>`;
      }
      const nq = r => r.query_terms.length;
      const hits = m => top(rs[0], m).filter(t => new RegExp('\\b' + s.mark + '\\b', 'i').test(t.title)).length;
      const marked = s.mark ? `<div class="sc-verdict"><span class="big">查「${esc(s.pair[0])}」，前 10 名標題寫著 ${esc(s.mark)}：${key('tfidf')}A ${hits('tfidf')} 篇　${key('w2v-tfidf')}B ${hits('w2v-tfidf')} 篇</span><span>${cos}</span></div>` : '';
      box.innerHTML = head(s, rs[0]) + marked + (s.mark ? '' : `
        <div class="sc-verdict"><span class="big">兩個查詢的前 10 名相同：${key('tfidf')}A ${shared('tfidf')} / 10　${key('w2v-tfidf')}B ${shared('w2v-tfidf')} / 10</span>
          <span>${sameTerms ? `前處理後都是 ${esc(rs[0].query_terms.join(', '))}` : cos}</span></div>`) + `
        <div class="sc-cols">${rs.map(r => `<div><div class="sc-q small">${esc(r.query)}</div>${words(r)}
          <table class="sc-strips"><tr><th>方法</th><th>前 10 名</th><th>P@10</th></tr>${r.methods.map(m => `<tr><td class="sc-mname">${key(m.method)}<b>${M[m.method].letter}</b>${modelTag(r.model, m.method)}</td><td class="sc-strip">${strip(m, nq(r))}</td><td>${m.measures ? f(m.measures['P@10']) : '–'}</td></tr>`).join('')}</table></div>`).join('')}</div>
        ${legend(!!rs[0].query_id)}
        ${titlesBox(`<div class="sc-cols">${rs.map(r => column(r.methods.find(m => m.method === 'w2v-tfidf'), '', nq(r), s.mark, 'B 查「' + esc(r.query) + '」')).join('')}</div>`, s)}`;
      bindTitles(box);
    } catch (e) { fail(box, e); }
  }

  /* ---------------- word level ---------------- */
  /* what the columns and tags of the word-level tables mean (the thresholds are the ones used below) */
  const RARE = 30, CTX_BOTH = 10, CTX_PMI = 1.5;
  const defs = pairs => `<div class="sc-defs"><h4>欄位與標記的定義</h4><dl>
    <dt>訓練次數</dt><dd>這個詞在 train（9,603 篇）中出現幾次，也就是它的向量被更新的機會。少於 ${RARE} 次標為 <span class="tag rare">稀有</span>，向量不穩定。</dd>
    <dt>同篇</dt><dd>兩個詞<b>同時出現</b>的 train 新聞篇數。</dd>
    <dt>PMI</dt><dd>PMI(a, b) = ln( 同篇 × N ÷ ( df<sub>a</sub> × df<sub>b</sub> ) )，N = 9,603，df = 含該詞的新聞篇數。0 = 和隨機碰在一起一樣；越大 = 越常一起出現（1.5 ≈ 隨機的 4.5 倍）。稀有詞的 PMI 容易虛高，要和「同篇」一起看（見範例「crudes vs barrel」）。</dd>
    ${pairs ? '' : `<dt><span class="tag ctx">情境共現</span></dt><dd>同篇 ≥ ${CTX_BOTH} 篇<b>而且</b> PMI ≥ ${CTX_PMI}：兩個詞常寫在同一篇新聞裡。它只說「常一起出現」，不能分辨是同義還是同一類新聞的常客。</dd>
    <dt>主題</dt><dd>含這個詞的 test 新聞中，五個主題裡佔比最高的那一個。鄰近詞的主題和查詢詞相同時標 <span class="tag topic">同主題</span>。主題是編目員貼的粗分類，共現是從文字統計的，兩者不同。</dd>`}</dl></div>`;

  async function wordShow(s, box) {
    const models = [st.sg, st.cbow].filter(Boolean);
    const bar = v => `<span class="sc-track"><i style="width:${Math.round(Math.max(0, v) * 100)}%"></i></span><span class="v">${f(v)}</span>`;
    const cnt = n => `${fmt(n)}${n < RARE ? '<span class="tag rare">稀有</span>' : ''}`;
    try {
      let h = `<div class="sc-head"><div class="sc-q">${esc(s.label)}</div><div class="sc-meta"><span class="chip">${esc(s.tag)}</span></div></div><p class="sc-note">${s.note}</p>`;
      if (s.pairs) {
        const rs = await Promise.all(models.map(m => api('/api/reuters/pairs?' + new URLSearchParams({ pairs: s.pairs, model: m }))));
        h += `<table class="sc-strips sc-pairs"><tr><th>詞對</th>${rs.map(r => `<th>cosine · ${esc(r.model)}</th>`).join('')}<th>訓練次數（左 ／ 右）</th><th>同篇</th><th>PMI</th></tr>
          ${rs[0].pairs.map((p, i) => `<tr><td class="sc-mname"><b>${esc(p.a)}</b> ／ <b>${esc(p.b)}</b></td>${rs.map(r => { const x = r.pairs[i]; return `<td class="sc-bar">${x.in_model ? bar(x.similarity) : '<span class="muted">OOV</span>'}</td>`; }).join('')}
            <td>${p.in_model && p.count_a != null ? cnt(p.count_a) + ' ／ ' + cnt(p.count_b) : '–'}</td><td>${p.in_model ? fmt(p.both) : '–'}</td><td>${p.in_model && p.pmi != null ? f(p.pmi, 1) : '–'}</td></tr>`).join('')}</table>`;
      } else {
        const r = await api('/api/reuters/similar?' + new URLSearchParams({ words: s.words, models: models.join(','), n: 5 }));
        h += r.words.map(w => {
          const own = (w.topic_profile.topics || []).filter(t => t.documents).sort((x, y) => y.share - x.share)[0];
          const count = (w.models.find(m => m.in_model) || {}).count;
          return `<div class="sc-word"><h4>${esc(w.word)} <span class="muted" title="出現在 ${fmt(w.df_train)} 篇 train 新聞（df）">訓練次數 <b>${count == null ? '–' : fmt(count)}</b>${count != null && count < RARE ? '<span class="tag rare">稀有</span>' : ''}${own ? ' · 主題 ' : ''}</span>${own ? tchip(own.topic) : ''}</h4><div class="sc-cols">
          ${w.models.map(mr => `<div class="sc-col"><h4>${esc(mr.model)}</h4>${mr.in_model ? `<table class="sc-strips sc-nb"><tr><th>鄰近詞</th><th>cosine</th><th>訓練次數</th><th>同篇</th><th>PMI</th><th>主題</th></tr>
            ${mr.neighbours.map(nb => `<tr><td class="sc-mname"><b>${esc(nb.word)}</b>${nb.both >= CTX_BOTH && nb.pmi != null && nb.pmi >= CTX_PMI ? '<span class="tag ctx">情境共現</span>' : ''}</td>
              <td class="sc-bar">${bar(nb.similarity)}</td><td>${cnt(nb.count)}</td><td>${fmt(nb.both)}</td><td>${nb.pmi == null ? '–' : f(nb.pmi, 1)}</td><td>${nb.top_topic ? tchip(nb.top_topic) + (own && nb.top_topic === own.topic ? '<span class="tag topic">同主題</span>' : '') : '–'}</td></tr>`).join('')}</table>` : '<p class="muted">OOV：不在詞彙中</p>'}</div>`).join('')}</div></div>`; }).join('');
      }
      box.innerHTML = h + defs(!!s.pairs);
    } catch (e) { fail(box, e); }
  }

  /* PMI is a ratio, not a count: the same PMI can rest on 12 documents or on 70 */
  const FEW = 20;
  async function pmiShow(s, box) {
    try {
      const rs = await Promise.all([st.sg, st.cbow].filter(Boolean).map(m => api('/api/reuters/pairs?' + new URLSearchParams({ pairs: s.pmi, model: m }))));
      const r = rs[0], N = 9603, rows = r.pairs.map((p, i) => ({ ...p, cos: rs.map(x => x.pairs[i].similarity) })).filter(p => p.in_model);
      const row = p => {
        const share = p.df_b ? p.both / p.df_b : 0, base = p.df_a / N;
        return `<tr><td class="sc-mname"><b>${esc(p.b)}</b> <span class="muted">和 ${esc(p.a)}</span></td>
          <td>${fmt(p.df_b)}</td><td><b>${fmt(p.both)}</b>${p.both && p.both < FEW ? '<span class="tag rare">樣本少</span>' : ''}</td>
          <td class="sc-bar"><span class="sc-track wide" data-tip="含 ${esc(p.b)} 的 ${fmt(p.df_b)} 篇中，${fmt(p.both)} 篇也有 ${esc(p.a)}"><i style="width:${Math.round(share * 100)}%"></i><u style="left:${(base * 100).toFixed(1)}%"></u></span><span class="v">${Math.round(share * 100)}%</span></td>
          <td>${(base * 100).toFixed(1)}%</td><td>${p.pmi == null ? '0' : '×' + Math.round(Math.exp(p.pmi))}</td><td><b>${p.pmi == null ? '–' : f(p.pmi, 2)}</b></td>${p.cos.map(c => `<td>${f(c)}</td>`).join('')}</tr>`;
      };
      box.innerHTML = `<div class="sc-head"><div class="sc-q">${esc(s.label)}</div><div class="sc-meta"><span class="chip">${esc(s.tag)}</span></div></div><p class="sc-note">${s.note}</p>
        <table class="sc-strips sc-pmi"><tr><th>詞 b（和 a）</th><th>b 出現在幾篇</th><th>同篇</th><th>b 的新聞中也有 a 的比例</th><th>隨機時的比例</th><th>高幾倍</th><th>PMI = ln(倍數)</th>${rs.map(x => `<th>cosine · ${esc(x.model.split('-')[0])}</th>`).join('')}</tr>${rows.map(row).join('')}</table>
        <div class="sc-legend"><span><i class="sc-key" style="background:var(--m-b)"></i>實際比例</span><span><i class="sc-key" style="background:var(--ink);width:2px"></i>隨機時的比例（a 在全部 ${fmt(N)} 篇 train 新聞中出現的比例）</span><span><span class="tag rare">樣本少</span>同篇少於 ${FEW} 篇</span></div>
        <div class="sc-defs"><h4>怎麼讀</h4><dl>
          <dt>PMI 說「多純」</dt><dd>有 b 的新聞裡，有多大比例也有 a，再除以隨機時的比例，取 ln。crudes 每一篇都有 crude（100%），所以 PMI 最高。</dd>
          <dt>同篇說「多穩」</dt><dd>這個比例建立在幾篇新聞上。12 篇時多一篇少一篇，PMI 就會大幅變動；71 篇穩定得多。</dd>
          <dt>結論</dt><dd>barrel 的 PMI 較低，卻是比 crudes 更強的證據。「情境共現」標記因此同時要求同篇 ≥ ${CTX_BOTH} 篇與 PMI ≥ ${CTX_PMI}。</dd>
          <dt>PMI = 0 或沒有</dt><dd>同篇 0 篇時 PMI 無法計算。CBOW 給 opec / coffee 的 cosine 很高，兩者卻從不同篇：向量相近來自句型相似（生產國、出口配額），不是主題相關。</dd></dl></div>`;
    } catch (e) { fail(box, e); }
  }

  async function run(id, custom) {
    const s = custom || ALL[id];
    if (!s) return;
    st.last[stepOf(s)] = s;
    go(stepOf(s), true);
    $$('#sc-buttons button').forEach(b => b.classList.toggle('on', b.dataset.sc === s.id));
    const box = document.createElement('div');           // a stale run keeps writing into its own detached box
    box.innerHTML = '<p class="muted">computing…</p>';
    $('#sc-out').replaceChildren(box);
    await st.ready;
    if (s.pair) await pairDuel(s, box); else if (s.pmi) await pmiShow(s, box); else if (s.words || s.pairs) await wordShow(s, box); else await duel(s, box);
  }

  function renderButtons() {
    const btn = s => `<button class="sc-btn${s.star ? ' star' : ''}" data-sc="${s.id}">${esc(s.label)}${s.tag ? `<small>${esc(s.tag)}</small>` : ''}</button>`;
    $('#sc-buttons').innerHTML = `<div class="sc-group" data-for="3"><div class="sc-btns">${WORDS.map(btn).join('')}</div></div>`
      + GROUPS.map(g => `<div class="sc-group" data-for="4">${g.title ? `<h4>${esc(g.title)}</h4>` : ''}<div class="sc-btns">${g.items.map(btn).join('')}</div></div>`).join('');
    $$('#sc-buttons button').forEach(b => b.addEventListener('click', () => run(b.dataset.sc)));
  }

  /* ---------------- the stepper ---------------- */
  const cardsOf = () => $$('#tab-emb .card[data-step]');
  function go(n, quiet) {
    n = Math.max(1, Math.min(STEPS.length, +n || 1));
    const changed = n !== st.step;
    st.step = n;
    try { localStorage.setItem('p4step', n); } catch (e) { /* private mode */ }
    const S = STEPS[n - 1];
    cardsOf().forEach(c => c.classList.toggle('p4-off', !c.dataset.step.split(' ').includes(String(n))));
    $$('#p4-steps button').forEach(b => { b.classList.toggle('on', +b.dataset.n === n); b.classList.toggle('done', +b.dataset.n < n); });
    $('#p4-ask').textContent = S.ask;
    $('#p4-pos').textContent = `${n} / ${STEPS.length} · ${S.name}`;
    $('#p4-prev').disabled = n === 1; $('#p4-next').disabled = n === STEPS.length;
    $('#emb-s0').dataset.mode = n;
    if (S.h) $('#sc-h').firstChild.nodeValue = S.h;
    if (!quiet && (n === 3 || n === 4)) run('', st.last[n] || ALL[n === 3 ? 'w-oil' : 'c03']);
    if (!quiet && changed) window.scrollTo({ top: 0 });
  }
  /* make an element inside the tab visible: switch to its step and open its card if it is a collapsed one */
  function reveal(sel) {
    const el = typeof sel === 'string' ? $(sel) : sel, card = el && el.closest('#tab-emb .card[data-step]');
    if (!card) return;
    const steps = card.dataset.step.split(' ');
    if (!steps.includes(String(st.step))) go(steps[0]);
    if (card.classList.contains('adv')) card.classList.add('open');
  }
  function renderSteps() {
    $('#p4-steps').innerHTML = STEPS.map(s => `<button data-n="${s.n}"><i>${s.n}</i>${esc(s.name)}</button>`).join('');
    $$('#p4-steps button').forEach(b => b.addEventListener('click', () => go(b.dataset.n)));
    $('#p4-prev').addEventListener('click', () => go(st.step - 1));
    $('#p4-next').addEventListener('click', () => go(st.step + 1));
    $$('#tab-emb .card').forEach(c => {
      const h = c.querySelector(':scope > h2'), hint = c.querySelector(':scope > p.hint');
      if (h && c.classList.contains('adv')) h.addEventListener('click', e => { if (!e.target.closest('button, a')) c.classList.toggle('open'); });
      if (h && hint) {
        const b = document.createElement('button');
        b.className = 'small p4-why'; b.textContent = '說明'; b.title = '顯示 / 隱藏這一節的詳細說明';
        b.addEventListener('click', e => { e.stopPropagation(); c.classList.toggle('show-hint'); if (c.classList.contains('adv')) c.classList.add('open'); });
        h.appendChild(b);
      }
    });
  }

  /* ---------------- experiment data ---------------- */
  const runOf = (m, model) => st.evalData.runs.find(s => s.method === m && (m === 'tfidf' || s.model === model));
  const MEAS = ['P@10', 'nDCG@10', 'MRR'];
  const VIEWS = [['kind', '依查詢用字'], ['query', '逐題'], ['model', '不同詞向量模型']];
  const perQ = (s, k) => s.per_query.map(q => q[k === 'MRR' ? 'RR' : k]);

  function tiles() {
    const A = runOf('tfidf'), B = runOf('w2v-tfidf', st.sg);
    if (!A || !B) return '';
    const d = k => { const a = A.mean[k], b = B.mean[k], p = (b - a) / (a || 1) * 100; return `<div class="tile sc-tile"><span>${k}</span><b>${f(a, 3)} <em>→</em> ${f(b, 3)}</b><span class="delta ${p >= 0 ? 'up' : 'down'}">${p >= 0 ? '▲ +' : '▼ '}${p.toFixed(0)}%</span></div>`; };
    const a = perQ(A, 'P@10'), b = perQ(B, 'P@10');
    const win = b.filter((v, i) => v > a[i]).length, lose = b.filter((v, i) => v < a[i]).length;
    const al = st.alpha, best = al && al.rows.find(x => x.alpha === al.best_alpha);
    return `<div class="tiles">${d('P@10')}${d('MRR')}${d('nDCG@10')}
      <div class="tile sc-tile"><span>逐題 P@10</span><b>B 勝 ${win} · A 勝 ${lose}</b><span class="delta">平手 ${a.length - win - lose}</span></div>
      ${best && st.show.hybrid ? `<div class="tile sc-tile"><span>融合 D 的最佳 α（自行加入）</span><b>α = ${al.best_alpha}</b><span class="delta">nDCG@10 ${f(best['nDCG@10'], 3)}</span></div>` : ''}</div>`;
  }

  function kindChart() {
    const k = st.measure, kinds = st.evalData.kinds;
    const runs = METHODS.filter(m => shown(m.id)).map(m => ({ m, s: runOf(m.id, st.sg) })).filter(x => x.s);
    const W = 760, H = 300, L = 44, R = 12, T = 22, Bm = 46, pw = W - L - R, ph = H - T - Bm;
    const gw = pw / kinds.length, bw = [40, 40, 32, 26][runs.length - 1] || 26, gap = 2, y = v => T + ph - v * ph;
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:${W}px" role="img" aria-label="${k} by query wording">`;
    [0, 0.25, 0.5, 0.75, 1].forEach(v => { s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v.toFixed(2)}</text>`; });
    kinds.forEach((kd, gi) => {
      const x0 = L + gi * gw + (gw - runs.length * (bw + gap) + gap) / 2;
      runs.forEach((x, i) => {
        const v = x.s.by_kind[kd.kind] ? x.s.by_kind[kd.kind][k] : 0, bx = x0 + i * (bw + gap), by = y(v), h = Math.max(0, y(0) - by), r = Math.min(4, h);
        s += `<path d="M${bx},${y(0)}V${by + r}q0,-${r} ${r},-${r}h${bw - 2 * r}q${r},0 ${r},${r}V${y(0)}z" fill="${x.m.v}" data-tip="<b>${x.m.letter} ${esc(x.m.name)}</b><br>${esc(KIND_ZH[kd.kind] || kd.name)}：${k} ${f(v, 3)}"/>`;
        if (x.m.id === 'tfidf' || x.m.id === 'w2v-tfidf') s += `<text x="${bx + bw / 2}" y="${by - 5}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--ink)">${f0(v)}</text>`;
      });
      s += `<text x="${L + gi * gw + gw / 2}" y="${H - Bm + 18}" text-anchor="middle" font-size="13" fill="var(--ink)">${esc(KIND_ZH[kd.kind] || kd.name)}</text>`;
    });
    s += `<text x="${L}" y="${H - 6}" font-size="11.5" fill="var(--muted)">← 查詢用字貼近新聞詞彙</text><text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="11.5" fill="var(--muted)">查詢用字遠離新聞詞彙 →</text></svg>`;
    return `<p class="sc-take">用字離新聞詞彙越遠，A 掉得越多，B 幾乎不動。</p><div class="sc-legend">${runs.map(x => `<span>${key(x.m.id)}${x.m.letter} ${esc(x.m.name)}</span>`).join('')}
      <span class="grow"></span>${extraPicks()}</div>${s}`;
  }

  function dumbbell() {
    const k = st.measure, A = runOf('tfidf'), B = runOf('w2v-tfidf', st.sg);
    const a = perQ(A, k), b = perQ(B, k);
    const rows = st.evalData.queries.map((q, i) => ({ q, a: a[i], b: b[i], d: b[i] - a[i] }));
    const mean = v => v.reduce((x, y) => x + y, 0) / (v.length || 1);
    const delta = d => (d > 0.0005 ? '▲ +' + f(d) : d < -0.0005 ? '▼ −' + f(-d) : '＝');
    const line = x => { const lo = Math.min(x.a, x.b), hi = Math.max(x.a, x.b), tie = x.a === x.b;
      return `<button class="sc-db-row" data-q="${esc(x.q.id)}" data-tip="<b>${esc(x.q.id)}</b> ${esc(x.q.text)}<br>${esc(KIND_ZH[x.q.kind] || '')} · ${esc(x.q.topic)} · 相關 ${fmt(x.q.relevant)} 篇<br>A ${f(x.a)} → B ${f(x.b)}">
        <span class="lab">${tchip(x.q.topic)}${st.sort === 'delta' ? `<span class="sc-kind k-${esc(x.q.kind)}">${esc(KIND_ZH[x.q.kind] || x.q.kind)}</span>` : ''}${esc(x.q.text)}</span>
        <span class="sc-db-track"><u style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></u><em class="a${tie ? ' tie' : ''}" style="left:${x.a * 100}%"></em><em class="b${tie ? ' tie' : ''}" style="left:${x.b * 100}%"></em></span>
        <span class="d">${delta(x.d)}</span></button>`; };
    let body;
    if (st.sort === 'delta') body = rows.slice().sort((x, y) => y.d - x.d).map(line).join('');
    else body = st.evalData.kinds.map((kd, gi) => {                // grouped by query wording, with the group mean on the header
      const g = rows.filter(x => x.q.kind === kd.kind), ma = mean(g.map(x => x.a)), mb = mean(g.map(x => x.b));
      const win = g.filter(x => x.d > 0).length, lose = g.filter(x => x.d < 0).length;
      return `<div class="sc-db-row group"><span class="lab"><b>${gi + 1}. ${esc(KIND_ZH[kd.kind] || kd.name)}</b><span class="muted">${esc(kd.description || '')}</span></span>
        <span class="sc-db-track"><u style="left:${Math.min(ma, mb) * 100}%;width:${Math.abs(mb - ma) * 100}%"></u><em class="a${Math.abs(ma - mb) < 0.0005 ? ' tie' : ''}" style="left:${ma * 100}%"></em><em class="b${Math.abs(ma - mb) < 0.0005 ? ' tie' : ''}" style="left:${mb * 100}%"></em></span>
        <span class="d" data-tip="這一類 ${g.length} 題的平均：A ${f(ma)} → B ${f(mb)}<br>B 勝 ${win} · A 勝 ${lose} · 平手 ${g.length - win - lose}">${delta(mb - ma)}</span></div>${g.map(line).join('')}`; }).join('');
    return `<p class="sc-take">${st.sort === 'delta' ? '依差距排序：最上面是 B 贏最多的題目。' : '依查詢用字分四類：越往下，用字離新聞越遠，B 的優勢越明顯。'}點一列看那一題的對決。</p>
      <div class="sc-legend"><span>${key('tfidf')}A TF-IDF</span><span>${key('w2v-tfidf')}B TF-IDF 加權詞向量</span><span class="muted">粗體列 = 該類 5 題的平均</span>
        <label class="check"><input type="checkbox" id="sc-sort"${st.sort === 'delta' ? ' checked' : ''}> 改依差距排序</label></div>
      <div class="sc-db"><div class="sc-db-row axis"><span></span><span class="sc-db-track"><i style="left:0%">0</i><i style="left:50%">0.5</i><i style="left:100%">1</i></span><span>B − A</span></div>${body}</div>`;
  }

  function modelChart() {
    const k = st.measure, A = runOf('tfidf');
    const rows = st.evalData.runs.filter(s => s.method === 'w2v-tfidf').map(s => ({ s, v: s.mean[k], info: st.models.find(m => m.name === s.model) || {} })).sort((x, y) => y.v - x.v);
    const a = A.mean[k];
    return `<p class="sc-take">同一個方法 B，換不同的詞向量模型；直線是 A（TF-IDF）。只有 20 題，0.05 以內的差距不算顯著。</p>
      <div class="sc-mb"><div class="sc-mb-row axis"><span></span><span class="sc-db-track"><i style="left:0%">0</i><i style="left:50%">0.5</i><i style="left:100%">1</i></span><span></span></div>
      ${rows.map(x => `<div class="sc-mb-row" data-tip="<b>${esc(x.s.model)}</b><br>${esc(x.info.model || '')} · dim ${x.info.dim} · window ${x.info.window} · ${x.info.epochs} epochs<br>${k} ${f(x.v, 3)}（A TF-IDF ${f(a, 3)}）">
        <span class="lab"><b>${esc(x.info.model === 'cbow' ? 'CBOW' : 'skip-gram')}</b> d${x.info.dim} · w${x.info.window} · ${x.info.epochs} ep</span>
        <span class="sc-db-track bar"><s style="width:${x.v * 100}%"></s><q style="left:${a * 100}%"></q></span><span class="d">${f(x.v, 3)}</span></div>`).join('')}
      <div class="sc-mb-row axis"><span></span><span class="sc-db-track"><i class="ref" style="left:${a * 100}%">A TF-IDF ${f(a, 3)}</i></span><span></span></div></div>`;
  }

  function renderData() {
    const box = $('#sc-data');
    if (!st.evalData) return;
    const seg = (cls, items, cur) => `<span class="sc-seg">${items.map(([k, label]) => `<button class="${cls}${k === cur ? ' on' : ''}" data-k="${k}"${cls === 'sc-view' ? ` data-view="${k}"` : ''}>${esc(label)}</button>`).join('')}</span>`;
    box.innerHTML = tiles() + `<div class="sc-bar2">${seg('sc-view', VIEWS, st.view)}<span class="grow"></span><span class="muted">指標</span>${seg('sc-meas', MEAS.map(k => [k, k]), st.measure)}</div>
      <div class="sc-viewbox">${st.view === 'kind' ? kindChart() : st.view === 'query' ? dumbbell() : modelChart()}</div>`;
    $$('.sc-meas', box).forEach(b => b.addEventListener('click', () => { st.measure = b.dataset.k; renderData(); }));
    $$('.sc-view', box).forEach(b => b.addEventListener('click', () => { st.view = b.dataset.k; renderData(); }));
    $$('input[data-extra]', box).forEach(i => i.addEventListener('change', () => { st.show[i.dataset.extra] = i.checked; renderData(); }));
    const sort = $('#sc-sort');
    if (sort) sort.addEventListener('change', e => { st.sort = e.target.checked ? 'delta' : 'kind'; renderData(); });
    $$('.sc-db-row[data-q]', box).forEach(b => b.addEventListener('click', () => {
      const q = st.evalData.queries.find(x => x.id === b.dataset.q);
      run(q.id, ALL[q.id] || { id: q.id, q: q.text, note: '' });
      window.scrollTo({ top: 0 });
    }));
  }

  /* ---------------- init ---------------- */
  async function load() {
    const mr = await api('/api/reuters/models');
    st.models = mr.models;
    const pick = arch => mr.models.find(m => m.model === arch && m.dim === 100 && m.window === 5 && m.epochs === 5);
    st.sg = (pick('skip-gram') || mr.models[0] || {}).name || '';
    st.cbow = (pick('cbow') || {}).name || '';
    $('#sc-model').innerHTML = mr.models.map(m => `<option value="${esc(m.name)}"${m.name === st.sg ? ' selected' : ''}>${esc(m.name)}</option>`).join('');
    $('#sc-topic').innerHTML = '<option value="">不判定</option>' + TOPICS.map(t => `<option value="${t}">${t}</option>`).join('');
  }
  async function loadData() {
    try {
      const [ev, al] = await Promise.all([
        api('/api/reuters/eval?' + new URLSearchParams({ methods: METHODS.map(m => m.id).join(','), models: st.models.map(m => m.name).join(','), k: 10, alpha: $('#sc-alpha').value })),
        api('/api/reuters/alpha?' + new URLSearchParams({ model: st.sg }))]);
      st.evalData = ev; st.alpha = al; st.queries = ev.queries;
      renderData();
    } catch (e) { $('#sc-data').innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; }
  }
  function init() {
    if (st.ready) return;
    renderButtons();
    st.ready = load().catch(e => { $('#sc-out').innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; });
    st.ready.then(() => { if (!st.models.length) { $('#sc-data').innerHTML = '<p class="muted">尚無詞向量模型：先在「詞向量」步驟訓練。</p>'; return; } loadData(); });
    let saved = 1;
    try { saved = +localStorage.getItem('p4step') || 1; } catch (e) { /* private mode */ }
    go(saved);
  }
  renderSteps();
  go(1, true);
  const free = () => { const q = $('#sc-q').value.trim(); if (q) run('', { id: '', q, topic: $('#sc-topic').value, note: '' }); };
  $('#sc-go').addEventListener('click', free);
  $('#sc-q').addEventListener('keydown', e => { if (e.key === 'Enter') free(); });
  document.addEventListener('tab', e => { if (e.detail === 'emb') init(); });
  if ($('#tab-emb').classList.contains('active')) init();
  window.IR.showcase = { run, go, reveal };
})();
