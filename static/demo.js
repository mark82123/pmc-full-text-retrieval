/* Classroom demo helpers: presentation zoom, card focus, cursor spotlight, QR code for students, scripted tour.
   Loaded after the other scripts; talks to them only through the DOM (tab buttons, inputs, buttons). */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const store = {
    get(k, d) { try { const v = localStorage.getItem('demo.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('demo.' + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };

  /* ---------------- the tour script ----------------
     tab    : tab to open          card : CSS selector (or "h2:text") of the card to scroll to / focus
     set    : {selector: value}    click: selector of a button to press after the inputs are set
     query  : run a search (Search tab)       note : what to say / what to look at (shown in the caption bar) */
  const STEPS = [
    { group: 'Project 1 · 全文檢索', title: '布林查詢與查詢分析', tab: 'search', query: 'covid AND vaccine NOT pregnancy',
      note: '看「Query analysis」：每個詞如何被小寫化、去停用詞、Porter 詞幹化（vaccine → vaccin），以及 df。NOT 用集合差集排除含 pregnancy 的文件。' },
    { group: 'Project 1 · 全文檢索', title: '片語查詢（positional index）', tab: 'search', query: '"gut microbiome"',
      note: '片語要求兩詞相鄰且順序相同；倒排索引記錄了詞的位置，snippet 中反白的是相鄰出現的位置。' },
    { group: 'Project 1 · 全文檢索', title: '前綴萬用字元', tab: 'search', query: 'immun*',
      note: 'immun* 先展開成索引中所有以 immun 開頭的詞幹（immun、immunotherapi、immunosuppress…），再用 OR 合併。看分析列的展開清單。' },
    { group: 'Project 1 · 全文檢索', title: '欄位限制查詢', tab: 'search', query: 'title:diabetes',
      note: '只比對落在標題區間的位置。可切換 Rank by = Abstract only 看分欄評分如何改變排序。' },
    { group: 'Project 1 · 全文檢索', title: '拼字校正（edit distance）', tab: 'search', query: 'semaglutde obesty',
      note: '兩個字都拼錯、查無結果 → 系統自動改用編輯距離最近、語料頻率最高的詞。橫幅顯示 d（距離）與 cf（頻率）。' },
    { group: 'Project 1 · 全文檢索', title: '模糊比對 ~', tab: 'search', query: 'tirzepatid~',
      note: 'tirzepatid~ 展開成字典中編輯距離 ≤ 2 的所有詞，等於「容錯的 OR 查詢」。' },
    { group: 'Project 1 · 全文檢索', title: 'BM25 vs TF-IDF 排序', tab: 'search', set: { '#method': 'tfidf' }, query: 'cancer immunotherapy',
      note: '同一查詢改用 TF-IDF 排序。再把 Scoring 切回 BM25 比較名次：BM25 的 tf 飽和與長度正規化讓長文件不再佔便宜。' },
    { group: 'Project 1 · 全文檢索', title: '文件檢視：句子邊界與反白', tab: 'search', set: { '#method': 'bm25' }, query: 'cancer immunotherapy', openFirst: true,
      note: '規則式 EOS 偵測（et al.、Fig. 2、0.05、E. coli 都不切句）；勾選「Only sentences with matches」只看命中句。上方統計方塊比較 rule-based 與 naive 句數。' },
    { group: 'Project 1 · 全文檢索', title: '語料庫統計', tab: 'corpus', card: '#corpus',
      note: '整個索引的字元數、token、不重複詞、句數；表格可依欄位排序，點一列開啟文件。' },

    { group: 'Project 1 · 前處理', title: '斷詞、小寫化與反例', tab: 'algo', card: 'h2:斷詞', click: '#analyze-btn',
      note: 'WHO → who 是停用詞被整個丟掉；SARS-CoV-2 被連字號切開；1,024 變成 1 與 024。這些都是真實行為，不是示意。' },
    { group: 'Project 1 · 前處理', title: '停用詞清單', tab: 'algo', card: 'h2:停用詞',
      note: 'Snowball 174 字 + 17 個生醫補充（et、al、however…）。SMART 571 字僅供比較。' },
    { group: 'Project 1 · 前處理', title: 'Porter 詞幹化逐步追蹤', tab: 'algo', card: 'h2:Porter', click: '#stem-btn',
      note: '每個字顯示五個步驟中哪一條規則被套用、measure m 的值；下方是 over-stemming / under-stemming 的真實反例。' },

    { group: 'Project 2 · Zipf', title: "Zipf's law：rank × frequency", tab: 'zipf', card: 'h2:Rank vs frequency',
      note: '最常見的詞 rank 1，頻率約為 rank 2 的兩倍、rank 3 的三倍。切換 A–E 五種前處理條件看曲線怎麼變。' },
    { group: 'Project 2 · Zipf', title: 'Log-log 圖與線性迴歸', tab: 'zipf', card: 'h2:Log-log',
      note: '取對數後近似直線，斜率約 −1。頭部（停用詞）與尾部（hapax）偏離直線的原因可以在這裡討論。' },
    { group: 'Project 2 · Zipf', title: 'Resolving power：有效詞區間', tab: 'zipf', card: 'h2:Resolving power',
      note: 'Luhn 的想法：太常見與太罕見的詞都沒有鑑別力，中間區段才是好的索引詞。上下 cut-off 可拖動。' },
    { group: 'Project 2 · Zipf', title: 'CF–DF 地圖', tab: 'zipf', card: 'h2:CF–DF',
      note: '功能詞、樣板詞、主題詞、關鍵詞落在不同區域：同樣的 CF，DF 低代表集中在少數文件（burstiness），是好的關鍵詞。' },

    { group: 'Project 2 · Word2Vec', title: '訓練樣本：Skip-gram vs CBOW', tab: 'w2v', card: '#w2v-s2', click: '#ex-go',
      note: '同一個句子，Skip-gram 產生 (中心詞 → 周圍詞) 配對、CBOW 產生 (周圍詞集合 → 中心詞)。改 window 看樣本數變化。' },
    { group: 'Project 2 · Word2Vec', title: '各模型的最近鄰比較', tab: 'w2v', card: '#w2v-s5', click: '#nn-go',
      note: '14 個詞逐模型列出前 5 名近鄰；粗體是所有模型共同找到的。注意藥名（semaglutide / tirzepatide / liraglutide）互為近鄰。' },
    { group: 'Project 2 · Word2Vec', title: '類比運算 a : b = c : ?', tab: 'w2v', card: 'h2:Analogy', click: '#w2v-ana-btn',
      note: 'v(ckd) − v(kidney) + v(liver) 最接近哪個詞？小語料的類比不一定成功，正好討論語料大小的影響。' },
    { group: 'Project 2 · Word2Vec', title: '語意配對的餘弦矩陣', tab: 'w2v', card: '#w2v-s6', click: '#pr-go',
      note: '組內（同類詞）的平均餘弦應高於組間。矩陣顏色越深越相似。' },
    { group: 'Project 2 · Word2Vec', title: 'PCA / t-SNE 視覺化', tab: 'w2v', card: '#w2v-s7', click: '#map-go',
      note: 'PCA 先出來，t-SNE 約需十幾秒。同類詞（藥名、器官、動物）聚成群；兩個 seed 的 t-SNE 形狀不同但群組一致。' },

    { group: 'Project 3 · TF-IDF', title: '詞項統計與 tf / idf 曲線', tab: 'vsm', card: '#vsm-s1', click: '#vt-go',
      note: '同一個詞的 CF、DF、IDF、最大 tf；下方曲線顯示 1 + ln tf 的飽和效果與 idf 隨 df 下降。' },
    { group: 'Project 3 · TF-IDF', title: '同一篇文件、不同 weighting', tab: 'vsm', card: '#vsm-s2', click: '#vd-go',
      note: 'nnn / lnn / ltc 等 SMART 記法：第一碼 tf、第二碼 df、第三碼正規化。看哪些詞的排名因為 idf 而上升或下降。' },
    { group: 'Project 3 · TF-IDF', title: '向量空間檢索', tab: 'vsm', card: '#vsm-s3', click: '#vq-go',
      note: '每篇結果列出各查詢詞的貢獻（w_q × w_d）。可切換文件端與查詢端 scheme。' },
    { group: 'Project 3 · TF-IDF', title: '對照實驗：一次改一個因素', tab: 'vsm', card: '#vsm-s4', click: '#vc-go',
      note: '同一查詢在多個 scheme 下的結果並排：重疊率、Kendall τ、文件長度平均。餘弦正規化（第三碼 c）對長文件偏差的影響最明顯。' },
    { group: 'Project 3 · TF-IDF', title: '評估：P@k、MAP、nDCG', tab: 'vsm', card: '#vsm-s5', click: '#ve-go',
      note: '以人工相關判斷計算各 scheme 的指標。可以點某個查詢看它的相關文件與排名。' },
    { group: 'Project 3 · TF-IDF', title: '研究問題（由即時數據計算）', tab: 'vsm', card: '#vsm-s6',
      note: 'RQ1–RQ5 的答案直接由目前的索引算出，不是寫死的文字。' },

    { group: 'Project 4 · Word Embedding 檢索', title: 'Reuters-21578 資料集', tab: 'emb', card: '#emb-s1',
      note: 'ModApte 切分：train / test 篇數、主題分布、test 詞彙與 word2vec 覆蓋率。' },
    { group: 'Project 4 · Word Embedding 檢索', title: 'Word-level：Top-5 近鄰', tab: 'emb', card: '#emb-s3', click: '#en-go',
      note: 'oil → crude / petroleum / barrel…。旁邊的共現次數與主題分布說明近鄰為什麼合理。' },
    { group: 'Project 4 · Word Embedding 檢索', title: 'Document-level：新聞檢索', tab: 'emb', card: '#emb-s4', click: '#eq-go',
      note: '四種文件表示法：TF-IDF、詞向量平均、TF-IDF 加權平均、混合。同一查詢切換 method 看結果差異。' },
    { group: 'Project 4 · Word Embedding 檢索', title: '同一查詢、不同表示法並排', tab: 'emb', card: '#emb-s5', click: '#ec-go',
      note: '詞向量方法能找到不含查詢詞但語意相近的新聞（查 oil 找到 petroleum），TF-IDF 則嚴格依字面比對。' },
    { group: 'Project 4 · Word Embedding 檢索', title: '評估：P@10、MRR、nDCG@10', tab: 'emb', card: '#emb-s6', click: '#ee-go',
      note: '20 個主題查詢的整體指標，並依查詢種類與主題拆開。混合法的 α 權重掃描在下方。' },
    { group: 'Project 4 · Word Embedding 檢索', title: '研究問題', tab: 'emb', card: '#emb-s7',
      note: 'RQ1–RQ5 由目前模型與索引即時計算。' },

    { group: '附錄 · Edit distance', title: '動態規劃表格', tab: 'spell', card: 'h2:Edit distance', click: '#ed-btn',
      note: 'DP 表每一格 = min(刪除、插入、取代)；回溯路徑顯示對齊方式。' },
    { group: '附錄 · Edit distance', title: '拼字校正候選詞', tab: 'spell', card: 'h2:Spelling correction', click: '#sp-btn',
      note: '字典中編輯距離 ≤ k 的詞，依距離再依語料頻率排序。這就是搜尋頁自動校正用的函式。' },
  ];

  /* ---------------- DOM: header button + panel + overlays ---------------- */
  const header = $('header.top');
  const btn = document.createElement('button');
  btn.id = 'demo-toggle'; btn.className = 'theme-btn demo-btn'; btn.title = '課堂展示工具 (快捷鍵 P)'; btn.textContent = '🎬 展示';
  header.appendChild(btn);

  const panel = document.createElement('aside');
  panel.id = 'demo-panel'; panel.className = 'demo-panel hidden';
  panel.innerHTML = `
    <div class="demo-head"><b>課堂展示工具</b><span class="muted">快捷鍵 <kbd>P</kbd> 開關</span><button class="small" id="demo-close" title="關閉">✕</button></div>
    <section>
      <h4>投影模式</h4>
      <div class="demo-row">
        <span>字級</span>
        ${[100, 115, 130, 150, 175].map(z => `<button class="small zoom" data-zoom="${z}">${z}%</button>`).join('')}
      </div>
      <label class="check"><input type="checkbox" id="demo-hide-sub"> 隱藏副標題與頁尾（畫面更簡潔）</label>
      <label class="check"><input type="checkbox" id="demo-autofocus"> 腳本每一步自動聚焦該卡片</label>
      <label class="check"><input type="checkbox" id="demo-spot-chk"> 游標聚光燈（<kbd>S</kbd>）</label>
      <p class="hint">滑鼠移到任一卡片標題，按右側的 <b>⤢</b> 只顯示那張卡片；<kbd>Esc</kbd> 離開。</p>
    </section>
    <section>
      <h4>學生連線 <button class="small" id="demo-qr-big">放大 QR</button></h4>
      <div id="demo-share" class="demo-share"><p class="muted">loading…</p></div>
    </section>
    <section class="demo-steps-sec">
      <h4>展示腳本 <span class="muted" id="demo-pos"></span></h4>
      <div class="demo-row">
        <button class="small" id="demo-prev" title="上一步 (←)">← 上一步</button>
        <button class="small primary" id="demo-next" title="下一步 (→)">下一步 →</button>
        <button class="small" id="demo-stop" title="離開腳本">結束</button>
        <label class="check"><input type="checkbox" id="demo-caption" checked> 顯示字幕</label>
      </div>
      <ol id="demo-steps" class="demo-steps"></ol>
    </section>
    <section>
      <h4>快捷鍵</h4>
      <table class="kv demo-keys">
        <tr><td><kbd>P</kbd></td><td>開關此面板</td></tr>
        <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>腳本上一步 / 下一步</td></tr>
        <tr><td><kbd>1</kbd>…<kbd>9</kbd>, <kbd>0</kbd></td><td>切換分頁（0 = 第 10 個）</td></tr>
        <tr><td><kbd>F</kbd></td><td>聚焦 / 取消聚焦滑鼠所在的卡片</td></tr>
        <tr><td><kbd>S</kbd></td><td>游標聚光燈</td></tr>
        <tr><td><kbd>Q</kbd></td><td>全螢幕 QR code</td></tr>
        <tr><td><kbd>+</kbd> / <kbd>-</kbd></td><td>放大 / 縮小字級</td></tr>
        <tr><td><kbd>Esc</kbd></td><td>離開聚焦 / 關閉 QR</td></tr>
      </table>
      <p class="hint">快捷鍵在游標不在輸入框時才有作用；先點一下頁面空白處即可。</p>
    </section>`;
  document.body.appendChild(panel);

  const caption = document.createElement('div');
  caption.id = 'demo-caption-bar'; caption.className = 'demo-caption hidden';
  document.body.appendChild(caption);

  const qrOverlay = document.createElement('div');
  qrOverlay.id = 'demo-qr-overlay'; qrOverlay.className = 'demo-overlay hidden';
  document.body.appendChild(qrOverlay);

  const spot = document.createElement('div');
  spot.id = 'demo-spot'; spot.className = 'demo-spot hidden';
  document.body.appendChild(spot);

  const focusBar = document.createElement('div');
  focusBar.id = 'demo-focus-bar'; focusBar.className = 'demo-focus-bar hidden';
  focusBar.innerHTML = '<span>聚焦模式：只顯示這張卡片</span><button class="small" id="demo-unfocus">顯示全部 (Esc)</button>';
  document.body.appendChild(focusBar);

  /* ---------------- panel open / close ---------------- */
  function togglePanel(force) {
    const open = force !== undefined ? force : panel.classList.contains('hidden');
    panel.classList.toggle('hidden', !open);
    document.body.classList.toggle('demo-open', open);
    if (open && !shareLoaded) loadShare();
  }
  btn.addEventListener('click', () => togglePanel());
  $('#demo-close', panel).addEventListener('click', () => togglePanel(false));

  /* ---------------- zoom / hide chrome ---------------- */
  function applyZoom(z) {
    document.body.style.zoom = z / 100;
    $$('.zoom', panel).forEach(b => b.classList.toggle('active', +b.dataset.zoom === z));
    store.set('zoom', z);
  }
  $$('.zoom', panel).forEach(b => b.addEventListener('click', () => applyZoom(+b.dataset.zoom)));
  applyZoom(store.get('zoom', 100));

  const hideSub = $('#demo-hide-sub', panel);
  hideSub.checked = store.get('hideSub', false);
  const applyHide = () => { document.body.classList.toggle('demo-clean', hideSub.checked); store.set('hideSub', hideSub.checked); };
  hideSub.addEventListener('change', applyHide); applyHide();

  const autoFocus = $('#demo-autofocus', panel);
  autoFocus.checked = store.get('autofocus', false);
  autoFocus.addEventListener('change', () => store.set('autofocus', autoFocus.checked));

  /* ---------------- spotlight ---------------- */
  const spotBox = $("#demo-spot-chk", panel);
  function setSpot(on) {
    spotBox.checked = on;
    spot.classList.toggle('hidden', !on);
    document.body.classList.toggle('demo-spot-on', on);
  }
  spotBox.addEventListener('change', () => setSpot(spotBox.checked));
  document.addEventListener('mousemove', e => {
    if (spot.classList.contains('hidden')) return;
    const z = parseFloat(document.body.style.zoom) || 1;      // body zoom scales client coordinates of fixed children
    spot.style.left = (e.clientX / z) + 'px'; spot.style.top = (e.clientY / z) + 'px';
  });
  document.addEventListener('mousedown', e => {
    if (spot.classList.contains('hidden')) return;
    spot.classList.remove('pulse'); void spot.offsetWidth; spot.classList.add('pulse');
  });

  /* ---------------- card focus ---------------- */
  function cardOf(el) { return el && el.closest ? el.closest('.card') : null; }
  function focusCard(card) {
    unfocus();
    if (!card) return;
    const p = card.closest('.panel') || document.body;
    p.classList.add('focus-mode');
    card.classList.add('focused');
    let up = card.parentElement;
    while (up && up !== p) { up.classList.add('focus-path'); up = up.parentElement; }
    focusBar.classList.remove('hidden');
    window.scrollTo(0, 0);
  }
  function unfocus() {
    $$('.focus-mode').forEach(p => p.classList.remove('focus-mode'));
    $$('.focused').forEach(c => c.classList.remove('focused'));
    $$('.focus-path').forEach(c => c.classList.remove('focus-path'));
    focusBar.classList.add('hidden');
  }
  $('#demo-unfocus', focusBar).addEventListener('click', unfocus);
  // a ⤢ button on every card heading (also for cards rendered later)
  function decorate(root = document) {
    $$('.card > h2', root).forEach(h => {
      if (h.querySelector('.focus-btn')) return;
      const b = document.createElement('button');
      b.className = 'focus-btn'; b.title = '只顯示這張卡片 (F)'; b.textContent = '⤢';
      b.addEventListener('click', e => { e.stopPropagation(); const c = cardOf(h); c.classList.contains('focused') ? unfocus() : focusCard(c); });
      h.appendChild(b);
    });
  }
  decorate();
  new MutationObserver(muts => { muts.forEach(m => m.addedNodes.forEach(n => { if (n.nodeType === 1) decorate(n); })); })
    .observe($('main'), { childList: true, subtree: true });
  // the card under the mouse / containing the focused element is the Alt+F target
  let hoverCard = null;
  document.addEventListener('mouseover', e => { const c = cardOf(e.target); if (c) hoverCard = c; });

  /* ---------------- share (LAN URL + QR) ---------------- */
  let shareLoaded = false, shareInfo = null;
  async function loadShare() {
    const box = $('#demo-share', panel);
    try {
      shareInfo = await (await fetch('/api/info')).json();
    } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    shareLoaded = true;
    const urls = shareInfo.urls && shareInfo.urls.length ? shareInfo.urls : [location.origin + '/'];
    const main = urls[0];
    let h = '';
    if (!shareInfo.bound_to_all) h += '<p class="warn">伺服器只綁定 127.0.0.1，學生無法連線。請以 <code>python3 app.py --host 0.0.0.0</code> 啟動。</p>';
    h += `<div class="demo-qr"><img src="/api/qr.svg?text=${encodeURIComponent(main)}" alt="QR"><div>`;
    h += urls.map((u, i) => `<div class="demo-url${i ? ' alt' : ''}"><a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a></div>`).join('');
    h += `<p class="hint">${shareInfo.documents != null ? shareInfo.documents.toLocaleString() + ' 篇文件 · ' : ''}${(shareInfo.terms || 0).toLocaleString()} 個詞項<br>手機與電腦需在同一個網路；WSL 需先在 Windows 設定 portproxy。</p></div></div>`;
    box.innerHTML = h;
  }
  function showQr(on) {
    if (on) {
      const urls = (shareInfo && shareInfo.urls && shareInfo.urls.length) ? shareInfo.urls : [location.origin + '/'];
      qrOverlay.innerHTML = `<div class="demo-overlay-inner"><img src="/api/qr.svg?text=${encodeURIComponent(urls[0])}" alt="QR">
        <div class="big-url">${esc(urls[0])}</div><p>掃描 QR code 或輸入網址，在自己的裝置上跟著操作 · 按 Esc 關閉</p></div>`;
      qrOverlay.classList.remove('hidden');
    } else qrOverlay.classList.add('hidden');
  }
  qrOverlay.addEventListener('click', () => showQr(false));
  $('#demo-qr-big', panel).addEventListener('click', async () => { if (!shareLoaded) await loadShare(); showQr(true); });

  /* ---------------- tour ---------------- */
  let cur = store.get('step', -1);
  const stepsEl = $('#demo-steps', panel);
  function renderSteps() {
    let h = '', lastGroup = null;
    STEPS.forEach((s, i) => {
      if (s.group !== lastGroup) { h += `<li class="group">${esc(s.group)}</li>`; lastGroup = s.group; }
      h += `<li class="step${i === cur ? ' current' : ''}${i < cur ? ' done' : ''}" data-i="${i}"><span class="n">${i + 1}</span>${esc(s.title)}</li>`;
    });
    stepsEl.innerHTML = h;
    $$('li.step', stepsEl).forEach(li => li.addEventListener('click', () => runStep(+li.dataset.i)));
    $('#demo-pos', panel).textContent = cur >= 0 ? `${cur + 1} / ${STEPS.length}` : `${STEPS.length} 步`;
    const c = $('li.current', stepsEl); if (c) c.scrollIntoView({ block: 'nearest' });
  }
  function showTab(name) { const t = $(`.tab[data-tab="${name}"]`); if (t) t.click(); }
  function findCard(sel) {
    if (!sel) return null;
    if (sel.startsWith('h2:')) {
      const txt = sel.slice(3);
      const h = $$('.panel.active .card > h2, .panel.active h2').find(x => x.textContent.includes(txt));
      return h ? (cardOf(h) || h.parentElement) : null;
    }
    const el = $(sel);
    return el ? (el.classList.contains('card') ? el : (cardOf(el) || el)) : null;
  }
  async function waitFor(sel, ms = 4000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const el = typeof sel === 'function' ? sel() : $(sel);
      if (el && !el.disabled) return el;
      await sleep(80);
    }
    return null;
  }
  function setCaption(step) {
    const on = $('#demo-caption', panel).checked && step;
    caption.classList.toggle('hidden', !on);
    document.body.classList.toggle('demo-caption-on', !!on);
    if (step) caption.innerHTML = `<span class="n">${cur + 1}</span><div><b>${esc(step.title)}</b><span>${esc(step.note)}</span></div>
      <span class="keys"><kbd>→</kbd> 下一步</span>`;
  }
  $('#demo-caption', panel).addEventListener('change', () => setCaption(cur >= 0 ? STEPS[cur] : null));

  async function runStep(i) {
    if (i < 0 || i >= STEPS.length) return;
    cur = i; store.set('step', cur);
    const s = STEPS[i];
    renderSteps(); setCaption(s);
    unfocus();
    showTab(s.tab);
    if (s.set) Object.entries(s.set).forEach(([sel, v]) => { const el = $(sel); if (el) { el.value = v; } });
    if (s.query) {
      const q = $('#q'); q.value = s.query;
      $('#search-form').requestSubmit();
      if (s.openFirst) {
        const a = await waitFor(() => $('#results a[data-doc]'));
        if (a) { await sleep(150); a.click(); }
      }
    }
    if (s.click) {
      const b = await waitFor(s.click);
      if (b) b.click();
    }
    await sleep(s.tab === 'zipf' ? 400 : 120);
    let card = findCard(s.card);
    if (!card && s.card && s.card.startsWith('h2:')) { await waitFor(() => findCard(s.card), 6000); card = findCard(s.card); }
    if (card) {
      if (autoFocus.checked && card.classList.contains('card')) focusCard(card);
      else card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      card.classList.add('demo-flash'); setTimeout(() => card.classList.remove('demo-flash'), 1600);
    } else if (!s.openFirst) window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function stopTour() { cur = -1; store.set('step', -1); renderSteps(); setCaption(null); unfocus(); }
  $('#demo-prev', panel).addEventListener('click', () => runStep(Math.max(0, cur - 1)));
  $('#demo-next', panel).addEventListener('click', () => runStep(Math.min(STEPS.length - 1, cur + 1)));
  $('#demo-stop', panel).addEventListener('click', stopTour);
  renderSteps();

  /* ---------------- keyboard (single keys, only when not typing in a field) ---------------- */
  const ZOOMS = [100, 115, 130, 150, 175];
  const typing = () => { const el = document.activeElement || {}; return /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || !!el.isContentEditable; };
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (!qrOverlay.classList.contains('hidden')) { showQr(false); return; }
      if ($('.focused')) { unfocus(); return; }
      return;
    }
    if (e.altKey || e.ctrlKey || e.metaKey || typing()) return;
    const k = e.key.toLowerCase();
    if (k === 'p') { e.preventDefault(); togglePanel(); }
    else if (k === 'arrowright' || k === 'pagedown') { e.preventDefault(); runStep(Math.min(STEPS.length - 1, cur + 1)); }
    else if (k === 'arrowleft' || k === 'pageup') { e.preventDefault(); runStep(Math.max(0, cur - 1)); }
    else if (k === 'f') {
      e.preventDefault();
      const f = $('.focused'); if (f) { unfocus(); return; }
      const c = hoverCard || $('.panel.active .card');
      if (c) focusCard(c);
    }
    else if (k === 's') { e.preventDefault(); setSpot(spot.classList.contains('hidden')); }
    else if (k === 'q') { e.preventDefault(); (shareLoaded ? Promise.resolve() : loadShare()).then(() => showQr(qrOverlay.classList.contains('hidden'))); }
    else if (k === '+' || k === '=') { e.preventDefault(); const z = store.get('zoom', 100); applyZoom(ZOOMS[Math.min(ZOOMS.length - 1, ZOOMS.indexOf(z) + 1)]); }
    else if (k === '-') { e.preventDefault(); const z = store.get('zoom', 100); applyZoom(ZOOMS[Math.max(0, ZOOMS.indexOf(z) - 1)]); }
    else if (/^[0-9]$/.test(k)) {
      const tabs = $$('.tab'); const t = tabs[k === '0' ? 9 : +k - 1]; if (t) { e.preventDefault(); t.click(); }
    }
  });

  // ?demo=1 or #...&demo opens the panel; ?step=N jumps to a step (handy for a bookmark on the podium PC)
  const params = new URLSearchParams(location.search);
  if (params.has('demo')) togglePanel(true);
  if (params.has('step')) setTimeout(() => runStep(+params.get('step') - 1), 300);
})();
