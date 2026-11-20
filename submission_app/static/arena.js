// Blind pairwise listening arena. Uses $, esc, api, heading and toast from app.js.
(() => {
const arrow = '<svg class="icon" aria-hidden="true" focusable="false"><use href="/static/icons.svg#arrow-up-right"></use></svg>';
const CHOICES = [
  ['a', 'A राम्रो', 'A sounds more natural'],
  ['b', 'B राम्रो', 'B sounds more natural'],
  ['both_good', 'दुवै प्राकृतिक', 'Both sound natural'],
  ['both_bad', 'कुनै पनि होइन', 'Neither sounds natural'],
];
let tags = [], regions = [];
// Keyboard: 1/2 play A/B, A/B/T/N choose, Enter submits, then Enter or → for the next pair.
let keyHandler = null;
document.addEventListener('keydown', (e) => {
  if (!keyHandler || e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (keyHandler(e.key.toLowerCase())) e.preventDefault();
});
window.addEventListener('hashchange', () => { keyHandler = null; });

const page = (body) => {
  $('#app').innerHTML = heading('Blind listening arena', 'Which voice sounds <em>more natural?</em>',
    'Hear the same Nepali sentence from two anonymous systems and choose. Names appear only after you vote.') + body;
};

async function arenaPage() {
  const s = await api('/api/arena/state');
  tags = s.tags;
  regions = s.regions || [];
  if (!s.ready) return page(`<div class="card empty"><div class="empty-symbol">A/B</div><h3>Not enough systems yet</h3><p>The arena opens once two approved systems allow anonymous listening.</p><a class="text-link" href="#human">See the human leaderboard ${arrow}</a></div>`);
  if (!s.listener) return intro();
  if (!s.listener.screened) return s.listener.attempts_left > 0 ? screeningIntro(s.listener.attempts_left)
    : page('<div class="notice red">The listening check could not be passed in three attempts, so votes from this browser are not collected. Thank you for trying.</div>');
  return battle(s.listener.votes);
}

function intro() {
  page(`<div class="two-col"><section class="card"><h3>Before you start</h3><ul class="checklist"><li>You will hear two clips of the same sentence, labelled A and B.</li><li>Play both to the end, then choose the one that sounds more like a real person speaking.</li><li>A short listening check comes first, so we know votes come from Nepali listeners.</li><li>Use headphones in a quiet place if you can.</li></ul>
  <label class="checkbox"><input type="checkbox" id="native"> म नेपाली मातृभाषी हुँ। I am a native Nepali speaker.</label>
  <label class="checkbox"><input type="checkbox" id="phones"> I am using headphones.</label>
  <label class="field"><span lang="ne">तपाईंको नेपाली कुन क्षेत्रको हो?</span> Where is your Nepali from? <small class="muted">Optional</small>
  <select id="region"><option value="">Prefer not to say</option>${regions.map(r => `<option value="${esc(r.id)}">${esc(r.ne)} · ${esc(r.en)}</option>`).join('')}</select></label>
  <button class="btn red" id="begin" disabled>Start listening ${arrow}</button></section>
  <aside><div class="card"><h3>What we store</h3><p>A random listener ID in a browser cookie, your answers, issue tags and response times. No name, email or account is needed. Votes are anonymous and used only to rank systems.</p></div></aside></div>`);
  $('#native').onchange = () => { $('#begin').disabled = !$('#native').checked; };
  $('#begin').onclick = async () => {
    $('#begin').disabled = true;
    const region = $('#region').value || null;
    try { await api('/api/arena/session', {method: 'POST', body: JSON.stringify({native_speaker: true, headphones: $('#phones').checked, region})}); await arenaPage(); }
    catch (e) { toast(e.message); $('#begin').disabled = false; }
  };
}

function screeningIntro(left) {
  page(`<section class="card"><h3>Listening check</h3><p>Play each clip and pick the sentence you hear. Three of four correct opens the arena. Attempts left: ${left}.</p><button class="btn red" id="screen-start">Begin the check ${arrow}</button></section>`);
  $('#screen-start').onclick = async () => {
    try { screening(await api('/api/arena/screening', {method: 'POST'})); }
    catch (e) { toast(e.message); }
  };
}

function screening(data) {
  page(`<form id="screen-form" class="stack">${data.items.map((item, i) => `<fieldset class="card arena-screen"><legend class="micro">CLIP ${i + 1} OF ${data.items.length}</legend><audio controls preload="auto" src="${esc(item.audio)}"></audio>${item.options.map((text, j) => `<label class="arena-option" lang="ne"><input type="radio" name="${esc(item.token)}" value="${j}" required> ${esc(text)}</label>`).join('')}</fieldset>`).join('')}<button class="btn red" type="submit">Check my answers ${arrow}</button></form>`);
  $('#screen-form').onsubmit = async (e) => {
    e.preventDefault();
    const answers = Object.fromEntries(data.items.map(item => [item.token, Number(new FormData(e.target).get(item.token))]));
    try {
      const r = await api('/api/arena/screening/answers', {method: 'POST', body: JSON.stringify({answers})});
      toast(r.passed ? 'Check passed. Welcome to the arena!' : `${r.correct} of ${data.items.length} correct. Attempts left: ${r.attempts_left}.`);
      await arenaPage();
    } catch (err) { toast(err.message); }
  };
}

async function battle(votes) {
  let b;
  try { b = await api('/api/arena/battles', {method: 'POST'}); }
  catch (e) { return page(`<div class="notice">${esc(e.message)}</div><a class="text-link" href="#human">See the human leaderboard ${arrow}</a>`); }
  const heard = {a: false, b: false};
  let choice = null;
  const done = Math.max(0, 50 - b.left_today - 1);
  page(`<div class="arena-meta micro"><span>YOUR VOTES · ${votes} · ${b.left_today} MORE TODAY</span><span class="arena-keys">KEYS · P both · 1 2 play · A B T N choose · ↵ submit</span></div>
  <div class="arena-progress" role="progressbar" aria-label="Today's comparisons" aria-valuemin="0" aria-valuemax="50" aria-valuenow="${done}"><i></i></div>
  <section class="card arena-prompt"><div class="eyebrow">PROMPT</div><p lang="ne" class="arena-text">${esc(b.text)}</p></section>
  <div class="arena-play-both"><button type="button" class="btn secondary small" id="play-both">▶ Play A, then B</button></div>
  <div class="arena-pair">${['a', 'b'].map(s => `<section class="card arena-side"><h3>Voice ${s.toUpperCase()}</h3><audio controls preload="auto" id="clip-${s}" src="${esc(b.audio[s])}"></audio><p class="micro" id="heard-${s}">Play to the end</p></section>`).join('')}</div>
  <section class="card"><p class="arena-question">Which sounds more natural, as though spoken by a real person?</p><div class="arena-choices">${CHOICES.map(([id, ne, en]) => `<button class="btn secondary" data-choice="${id}" disabled><span lang="ne">${ne}</span><small>${en}</small></button>`).join('')}</div>
  <div id="tag-panel" hidden><p class="micro">OPTIONAL · WHAT WENT WRONG?</p><div class="arena-tags">${['a', 'b'].map(s => `<div><b>Voice ${s.toUpperCase()}</b>${tags.map(t => `<button type="button" class="chip arena-tag" data-side="${s}" data-tag="${esc(t.id)}" aria-pressed="false"><span lang="ne">${esc(t.ne)}</span> · ${esc(t.en)}</button>`).join('')}</div>`).join('')}</div><button class="btn red" id="submit-vote">Submit and reveal ${arrow}</button></div>
  <div id="reveal" hidden></div></section>`);
  ['a', 'b'].forEach(s => {
    $('#clip-' + s).onplay = () => { const other = $('#clip-' + (s === 'a' ? 'b' : 'a')); if (!other.paused) other.pause(); };
    $('#clip-' + s).onended = () => {
      heard[s] = true;
      $('#heard-' + s).textContent = 'Heard ✓';
      if (heard.a && heard.b) document.querySelectorAll('[data-choice]').forEach(x => { x.disabled = false; });
    };
  });
  document.querySelectorAll('[data-choice]').forEach(btn => btn.onclick = () => {
    choice = btn.dataset.choice;
    document.querySelectorAll('[data-choice]').forEach(x => x.classList.toggle('active', x === btn));
    $('#tag-panel').hidden = false;
  });
  $('.arena-progress i').style.width = `${Math.min(100, done * 2)}%`;  // CSSOM, not an inline style: CSP allows it
  const play = (s) => { const el = $('#clip-' + s); el.currentTime = 0; el.play(); };
  const playBoth = () => { const a = $('#clip-a'); a.currentTime = 0; a.play(); a.addEventListener('ended', () => play('b'), {once: true}); };
  $('#play-both').onclick = playBoth;
  const pick = (id) => { const btn = document.querySelector(`[data-choice="${id}"]`); if (btn && !btn.disabled) btn.click(); return !!btn; };
  keyHandler = (key) => {
    if (!$('#reveal').hidden) { if (key === 'enter' || key === 'arrowright') { $('#next').click(); return true; } return false; }
    if (key === '1') { play('a'); return true; }
    if (key === '2') { play('b'); return true; }
    if (key === 'p') { playBoth(); return true; }
    const choice = {a: 'a', b: 'b', t: 'both_good', n: 'both_bad'}[key];
    if (choice) return pick(choice);
    if (key === 'enter' && choice === undefined && !$('#submit-vote').disabled && document.querySelector('[data-choice].active')) { $('#submit-vote').click(); return true; }
    return false;
  };
  document.querySelectorAll('.arena-tag').forEach(t => t.onclick = () => {
    const on = t.getAttribute('aria-pressed') !== 'true';
    t.setAttribute('aria-pressed', String(on));
    t.classList.toggle('red', on);
  });
  $('#submit-vote').onclick = async () => {
    const picked = {a: [], b: []};
    document.querySelectorAll('.arena-tag[aria-pressed="true"]').forEach(t => picked[t.dataset.side].push(t.dataset.tag));
    $('#submit-vote').disabled = true;
    try {
      const r = await api(`/api/arena/battles/${encodeURIComponent(b.token)}/vote`, {method: 'POST', body: JSON.stringify({choice, tags: picked})});
      document.querySelectorAll('[data-choice],.arena-tag').forEach(x => { x.disabled = true; });
      $('#tag-panel').hidden = true;
      const name = v => `${esc(v.system)}${v.organization ? ` <small>${esc(v.organization)}</small>` : ''}`;
      $('#reveal').innerHTML = `<div class="notice green"><b>A</b> · ${name(r.reveal.a)}<br><b>B</b> · ${name(r.reveal.b)}${r.note ? `<br>${esc(r.note)}` : ''}</div><button class="btn red" id="next">Next pair ${arrow}</button>`;
      $('#reveal').hidden = false;
      $('#next').onclick = () => battle(votes + 1);
      $('#next').focus();
    } catch (e) { toast(e.message); $('#submit-vote').disabled = false; }
  };
}

async function humanBoard() {
  const d = await api('/api/arena/leaderboard');
  const label = Object.fromEntries((d.tags || []).map(t => [t.id, t.en]));
  const kind = e => e.kind ? ` <span class="chip ${e.kind === 'human_reference' ? 'green' : ''}">${esc(d.kinds[e.kind])}</span>` : '';
  const sample = e => `<button type="button" class="btn secondary small arena-sample" data-sample="${esc(e.id)}" aria-label="Play a sample from ${esc(e.system)}">▶</button>`;
  const rows = d.entries.map(e => `<tr><td class="rank">${String(e.rank).padStart(2, '0')}</td><td class="system-cell">${sample(e)} ${esc(e.system)}${kind(e)}<small>${esc(e.organization)} · ${esc(e.voice)}</small></td><td><span class="score">${Math.round(e.elo)}</span>${e.provisional ? ' <span class="chip amber">Provisional</span>' : ''}</td><td>${e.ci95 ? `[${Math.round(e.ci95[0])}, ${Math.round(e.ci95[1])}]` : '–'}</td><td>${e.rank_range ? (e.rank_range[0] === e.rank_range[1] ? e.rank_range[0] : `${e.rank_range[0]}–${e.rank_range[1]}`) : '–'}</td><td>${e.votes}</td><td>${e.win_rate == null ? '–' : Math.round(e.win_rate * 100) + '%'}</td><td>${Object.entries(e.issues).map(([t, share]) => `<span class="chip">${esc(label[t] || t)} ${Math.round(share * 100)}%</span>`).join(' ')}</td></tr>`).join('');
  $('#app').innerHTML = heading('Human preference · blind pairwise', 'What Nepali <em>listeners</em> prefer.',
    'Native listeners compare two anonymous systems on the same sentence. Ratings use the Bradley-Terry model on the Elo scale.') +
    `<div class="section-heading"><span class="micro">${d.votes} COUNTED VOTES · ${d.listeners} LISTENERS</span><a class="btn red small" href="#vote">Listen and vote ${arrow}</a></div>` +
    (d.entries.length ? `<div class="table-wrap"><table><thead><tr><th>#</th><th>System</th><th>Elo ↑</th><th>95% CI</th><th>Rank range</th><th>Votes</th><th>Win rate</th><th>Top issues</th></tr></thead><tbody>${rows}</tbody></table></div><div class="table-footer"><span>OVERLAPPING RANK RANGES MEAN A STATISTICAL TIE</span><span>PROVISIONAL BELOW 30 VOTES</span></div>`
      : `<div class="card empty"><div class="empty-symbol">A/B</div><h3>No counted votes yet</h3><p>Be one of the first listeners.</p><a class="btn red" href="#vote">Start listening ${arrow}</a></div>`) +
    categoryTable(d) +
    `<div class="notice">${esc(d.method)}</div>` +
    `<p class="micro">UPDATED ${esc(new Date(d.updated * 1000).toLocaleString())} · SAMPLES READ: <span lang="ne">${esc(d.sample_sentence.text)}</span>${d.export_public ? ` · <a class="text-link" href="/api/arena/votes.csv">Download counted votes (CSV)</a>` : ''}</p>`;
  let player = null;
  document.querySelectorAll('[data-sample]').forEach(btn => btn.onclick = () => {
    if (player) player.pause();
    player = new Audio(`/api/arena/samples/${encodeURIComponent(btn.dataset.sample)}`);
    player.play().catch(() => toast('Sample unavailable.'));
  });
}

const CATEGORY = {phonological_minimal_pairs: 'Minimal pairs', question_intonation: 'Questions', emotion: 'Emotion',
  contrastive_stress: 'Stress', homographs: 'Homographs', newspaper_ambiguities: 'News', phrases: 'Phrases', robustness: 'Robustness'};
function categoryTable(d) {
  const cats = Object.entries(d.categories || {});
  if (!cats.length || !d.entries.length) return '';
  const head = cats.map(([c, v]) => `<th>${esc(CATEGORY[c] || c)}<small> · ${v.votes}</small></th>`).join('');
  const body = d.entries.map(e => `<tr><td class="system-cell">${esc(e.system)}</td>${cats.map(([, v]) => `<td>${v.elo[e.id] == null ? '–' : Math.round(v.elo[e.id])}</td>`).join('')}</tr>`).join('');
  return `<h3 class="arena-subhead">By prompt type</h3><div class="table-wrap"><table><thead><tr><th>System</th>${head}</tr></thead><tbody>${body}</tbody></table></div><p class="micro">${esc(d.category_note)}</p>`;
}

window.consentCard = (d, id, token) => d.state !== 'published' ? '' :
  `<div class="card"><h3>Blind listening arena</h3><p>${d.listening_consent ? 'Your approved audio can be played, anonymised and loudness-matched, to listeners comparing systems.' : 'Allow your approved audio to be played anonymously to listeners, so your system gets a human-preference rating.'}</p><button class="btn secondary small" data-consent="${d.listening_consent ? 'false' : 'true'}" data-sid="${esc(id)}" data-token="${esc(token || '')}">${d.listening_consent ? 'Withdraw from listening' : 'Allow anonymous listening'}</button><p class="micro">Withdrawing stops new comparisons; votes already cast are kept.</p></div>`;

document.addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-consent]');
  if (!btn) return;
  btn.disabled = true;
  try {
    await api(`/api/submissions/${encodeURIComponent(btn.dataset.sid)}/listening-consent`, {method: 'POST', body: JSON.stringify({granted: btn.dataset.consent === 'true'}), headers: btn.dataset.token ? {'X-Submission-Token': btn.dataset.token} : {}});
    toast(btn.dataset.consent === 'true' ? 'Anonymous listening allowed.' : 'Listening withdrawn.');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  } catch (err) { toast(err.message); btn.disabled = false; }
});

window.arenaPage = arenaPage;
window.humanBoard = humanBoard;
})();
