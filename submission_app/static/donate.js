// Voice donation: consent, profile, mic check, recording, donor dashboard, reviewer queue.
// Uses $, esc, api, heading and toast from app.js.
(() => {
const arrow = '<svg class="icon" aria-hidden="true" focusable="false"><use href="/static/icons.svg#arrow-up-right"></use></svg>';
const MAX_SECONDS = 20;
const OPTIONS = {
  age_band: [['18-24', '18–24'], ['25-34', '25–34'], ['35-44', '35–44'], ['45-54', '45–54'], ['55-64', '55–64'], ['65+', '65+']],
  gender: [['female', 'Female · महिला'], ['male', 'Male · पुरुष'], ['other', 'Other · अन्य'], ['prefer_not', 'Prefer not to say']],
  province: [['koshi', 'Koshi'], ['madhesh', 'Madhesh'], ['bagmati', 'Bagmati'], ['gandaki', 'Gandaki'], ['lumbini', 'Lumbini'], ['karnali', 'Karnali'], ['sudurpashchim', 'Sudurpashchim'], ['outside_nepal', 'Outside Nepal'], ['prefer_not', 'Prefer not to say']],
  first_language: [['nepali', 'Nepali · नेपाली'], ['maithili', 'Maithili'], ['bhojpuri', 'Bhojpuri'], ['tharu', 'Tharu'], ['tamang', 'Tamang'], ['newar', 'Newar · नेपाल भाषा'], ['magar', 'Magar'], ['bajjika', 'Bajjika'], ['doteli', 'Doteli'], ['other', 'Other']],
  device: [['phone', 'Phone'], ['laptop', 'Laptop microphone'], ['headset', 'Headset'], ['external_mic', 'External microphone'], ['other', 'Other']],
};
const STATE = {asr_pending: ['Checking', ''], accepted: ['Accepted', 'green'], needs_review: ['In review', 'amber'], rejected: ['Not accepted', 'red']};

// ------------------------------------------------------------------ microphone

const mic = {stream: null, ctx: null, chunks: [], recording: false, timer: null, onLevel: null};

async function openMic() {
  if (mic.ctx) return;
  mic.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
  mic.ctx = new AudioContext();
  await mic.ctx.audioWorklet.addModule('/static/recorder-worklet.js');
  const node = new AudioWorkletNode(mic.ctx, 'pcm-capture');
  const mute = mic.ctx.createGain();
  mute.gain.value = 0; // keeps the graph running without playing the mic back
  mic.ctx.createMediaStreamSource(mic.stream).connect(node).connect(mute).connect(mic.ctx.destination);
  node.port.onmessage = (e) => {
    if (!mic.recording) return;
    mic.chunks.push(e.data);
    if (mic.onLevel) { let s = 0; for (const v of e.data) s += v * v; mic.onLevel(Math.sqrt(s / e.data.length)); }
  };
}

function closeMic() {
  clearTimeout(mic.timer);
  mic.recording = false;
  if (mic.stream) mic.stream.getTracks().forEach(t => t.stop());
  if (mic.ctx) mic.ctx.close();
  Object.assign(mic, {stream: null, ctx: null, chunks: []});
}
window.addEventListener('hashchange', closeMic);

function startRecording(onStop) {
  mic.chunks = [];
  mic.recording = true;
  mic.timer = setTimeout(onStop, MAX_SECONDS * 1000 - 200);
}

function stopRecording() {
  clearTimeout(mic.timer);
  mic.recording = false;
  let rate = mic.ctx.sampleRate;
  let samples = new Float32Array(mic.chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of mic.chunks) { samples.set(c, offset); offset += c.length; }
  while (rate > 48000) { // e.g. 96 kHz interfaces: halve by averaging pairs
    const half = new Float32Array(samples.length >> 1);
    for (let i = 0; i < half.length; i++) half[i] = (samples[2 * i] + samples[2 * i + 1]) / 2;
    samples = half;
    rate /= 2;
  }
  return encodeWav(samples, rate);
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buffer);
  const text = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  text(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE');
  text(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  text(36, 'data'); v.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
  return new Blob([buffer], {type: 'audio/wav'});
}

const upload = (url, blob) => api(url, {method: 'POST', body: blob, headers: {'Content-Type': 'audio/wav'}});
const meter = () => `<div class="donate-meter" aria-hidden="true"><span id="level"></span></div>`;
const showLevel = (rms) => { const el = $('#level'); if (el) el.style.width = Math.min(100, Math.max(0, (20 * Math.log10(rms + 1e-9) + 60) / 60 * 100)) + '%'; };

// ------------------------------------------------------------------ pages

const options = (name) => `<select name="${name}" required><option value="">Choose…</option>${OPTIONS[name].map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('')}</select>`;

async function donatePage() {
  const info = await api('/api/donate/info');
  if (info.me) { location.hash = 'record'; return; }
  const c = info.consent;
  const paid = info.paid_programme.open ? `Paid contributors earn NPR ${info.paid_programme.rate_npr_per_minute} per accepted minute; apply after joining.` : 'A paid programme will open later.';
  $('#app').innerHTML = heading('Voice donation · आवाज दान', 'Lend your voice to <em>Nepali</em> speech.',
    `Read Nepali sentences aloud in your browser. Your recordings become the human reference that every Nepali speech system is measured against. ${esc(paid)}`) +
    (info.open ? '' : `<div class="notice red"><b>Not open yet.</b> ${esc(c.review_note || 'The consent text is awaiting approval.')}</div>`) +
    `<div class="two-col"><form class="card stack" id="enroll"><div><div class="eyebrow">${esc(c.version)}</div><h3>${esc(c.title.en)} · <span lang="ne">${esc(c.title.ne)}</span></h3></div>
    ${c.sections.map(s => `<div class="donate-consent"><p lang="ne">${esc(s.ne)}</p><p>${esc(s.en)}</p></div>`).join('')}
    <h3>What may we use your recordings for?</h3>
    ${c.purposes.map(p => `<label class="checkbox"><input type="checkbox" name="purpose" value="${esc(p.id)}" ${p.required ? 'checked disabled' : ''}><span><span lang="ne">${esc(p.ne)}</span><br>${esc(p.en)}${p.required ? ' <span class="chip">Required</span>' : ''}${p.open_sets_only ? ' <span class="chip">Open sentences only</span>' : ''}</span></label>`).join('')}
    <h3>About you <span class="micro">(for balanced coverage; no name or contact needed)</span></h3>
    <div class="donate-grid">${[['age_band', 'Age'], ['gender', 'Gender'], ['province', 'Province'], ['first_language', 'First language'], ['device', 'Recording device']].map(([n, l]) => `<label class="field">${l}${options(n)}</label>`).join('')}</div>
    <label class="checkbox"><input type="checkbox" name="adult" required> I am 18 or older. म १८ वर्ष वा माथिको हुँ।</label>
    <label class="checkbox"><input type="checkbox" name="own_voice" required> I will record only my own voice.</label>
    <button class="btn red" ${info.open ? '' : 'disabled'}>Agree and start ${arrow}</button></form>
    <aside><div class="card"><h3>Already joined?</h3><p>Enter your recovery code to continue on this device.</p><form id="recover"><label class="field">Recovery code<input name="code" autocomplete="off" required></label><button class="btn secondary small">Recover</button></form></div>
    <div class="card"><h3>Where recordings go</h3><p>${info.prompt_sets.map(s => `<b>${esc(s.id)}</b> · ${s.count} sentences. ${esc(s.description)}`).join('<br>')}</p></div></aside></div>`;
  $('#recover').onsubmit = async (e) => {
    e.preventDefault();
    try { await api('/api/donate/recover', {method: 'POST', body: JSON.stringify({code: new FormData(e.target).get('code')})}); location.hash = 'record'; }
    catch (err) { toast(err.message); }
  };
  $('#enroll').onsubmit = async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const body = {consent_version: c.version, consent_sha256: c.sha256, adult: true, own_voice: true,
      purposes: [...e.target.querySelectorAll('[name=purpose]:checked')].map(x => x.value)};
    for (const n of Object.keys(OPTIONS)) body[n] = f.get(n);
    try {
      const r = await api('/api/donate/enroll', {method: 'POST', body: JSON.stringify(body)});
      $('#app').innerHTML = heading('Welcome', 'Save your <em>recovery code.</em>', 'It is the only way to reach your recordings from another browser or after clearing cookies. We cannot show it again.') +
        `<div class="card"><p class="donate-code">${esc(r.recovery_code)}</p><button class="btn secondary small" id="copy-code">Copy code</button> <a class="btn red" href="#record">I saved it · start recording ${arrow}</a></div>`;
      $('#copy-code').onclick = async () => { try { await navigator.clipboard.writeText(r.recovery_code); toast('Recovery code copied'); } catch { toast('Copy the code by hand.'); } };
    } catch (err) { toast(err.message); }
  };
}

async function recordPage() {
  let me;
  try { me = await api('/api/donate/me'); } catch { location.hash = 'donate'; return; }
  if (!me.mic_check || !me.mic_check.passed) return micCheck();
  const next = await api('/api/donate/next');
  if (next.done) {
    $('#app').innerHTML = heading('All done', 'You have read <em>every sentence.</em>', 'Thank you. Accepted recordings appear on your page.') + `<a class="btn red" href="#my-voice">See my recordings ${arrow}</a>`;
    return;
  }
  $('#app').innerHTML = heading('Record', 'Read the sentence <em>naturally.</em>', 'Speak as you normally would, at a calm pace. Pause for a moment before you start.') +
    `<div class="donate-stats micro">${me.counts.accepted} ACCEPTED · ${me.counts.asr_pending + me.counts.needs_review} CHECKING · ${next.remaining} LEFT · <a class="text-link" href="#my-voice">My recordings</a></div>
    <section class="card donate-card"><div class="eyebrow">${esc(next.prompt_set)} · ${esc(next.prompt_id)}</div><p class="donate-text" lang="ne">${esc(next.text)}</p>${meter()}
    <div class="review-actions"><button class="btn red" id="rec">● Record</button><button class="btn secondary" id="skip">Skip sentence</button></div>
    <div id="preview" hidden><audio controls id="take"></audio><div class="review-actions"><button class="btn red" id="send">Submit ${arrow}</button><button class="btn secondary" id="again">Record again</button></div></div>
    <div id="feedback"></div></section>`;
  let blob = null;
  mic.onLevel = showLevel;
  const stop = () => {
    blob = stopRecording();
    $('#rec').textContent = '● Record';
    $('#take').src = URL.createObjectURL(blob);
    $('#preview').hidden = false;
  };
  $('#rec').onclick = async () => {
    if (mic.recording) return stop();
    try { await openMic(); } catch { return toast('Microphone access is needed to record.'); }
    $('#preview').hidden = true;
    $('#feedback').innerHTML = '';
    startRecording(stop);
    $('#rec').textContent = '■ Stop';
  };
  $('#again').onclick = () => { $('#preview').hidden = true; $('#rec').click(); };
  $('#skip').onclick = () => recordPage();
  $('#send').onclick = async () => {
    $('#send').disabled = true;
    try {
      await upload(`/api/donate/recordings/${encodeURIComponent(next.prompt_set)}/${encodeURIComponent(next.prompt_id)}`, blob);
      toast('Saved. धन्यवाद!');
      await recordPage();
    } catch (err) {
      $('#feedback').innerHTML = `<div class="error" role="alert">${esc(err.message)}</div>`;
      $('#send').disabled = false;
    }
  };
}

function micCheck() {
  $('#app').innerHTML = heading('Microphone check', 'Let’s hear your <em>room.</em>', 'Stay quiet for two seconds, then say: “नमस्ते, म नेपाली बोल्छु।” We measure loudness and background noise. Nothing is stored.') +
    `<section class="card donate-card">${meter()}<div class="review-actions"><button class="btn red" id="check">● Start 6-second check</button></div><div id="result"></div></section>`;
  mic.onLevel = showLevel;
  $('#check').onclick = async () => {
    try { await openMic(); } catch { return toast('Microphone access is needed to record.'); }
    $('#check').disabled = true;
    $('#check').textContent = 'Listening…';
    startRecording(() => {});
    setTimeout(async () => {
      const r = await api('/api/donate/mic-check', {method: 'POST', body: stopRecording(), headers: {'Content-Type': 'audio/wav'}}).catch(e => ({messages: [e.message]}));
      const m = r.metrics || {};
      $('#result').innerHTML = r.passed
        ? `<div class="notice green">Sounds good: speech ${m.snr_db} dB above the background.</div><button class="btn red" id="go-record">Start recording ${arrow}</button>`
        : `<div class="notice red">${(r.messages || []).map(esc).join(' ')}${m.snr_db != null ? ` (speech ${m.snr_db} dB above background; 15 dB needed)` : ''}</div>`;
      $('#check').disabled = false;
      $('#check').textContent = '● Try again';
      if (r.passed) $('#go-record').onclick = () => recordPage();
    }, 6000);
  };
}

async function myVoicePage() {
  let me, info;
  try { [me, info] = await Promise.all([api('/api/donate/me'), api('/api/donate/info')]); } catch { location.hash = 'donate'; return; }
  const paid = me.paid_status === 'approved'
    ? `<div class="stats"><div class="stat"><strong>${me.accepted_minutes}</strong><span>accepted minutes</span></div><div class="stat"><strong>NPR ${me.earned_npr}</strong><span>earned at ${me.rate_npr_per_minute}/min</span></div><div class="stat"><strong>NPR ${me.paid_npr}</strong><span>paid</span></div><div class="stat"><strong>NPR ${me.balance_npr}</strong><span>balance</span></div></div>`
    : me.paid_status === 'applied' ? '<p>Your paid-programme application is being reviewed.</p>'
    : me.paid_status === 'rejected' ? '<p>Your paid-programme application was not accepted. You can keep contributing as a volunteer.</p>'
    : info.paid_programme.open ? `<form id="apply" class="donate-grid"><label class="field">Payout method<select name="method"><option value="esewa">eSewa</option><option value="khalti">Khalti</option><option value="bank">Bank transfer</option></select></label><label class="field">Wallet number or account<input name="handle" required minlength="5"></label><label class="field">Name on the account<input name="name" required minlength="2"></label><div><button class="btn red small">Apply · NPR ${info.paid_programme.rate_npr_per_minute}/accepted minute</button></div></form><p class="micro">Only reviewers see payout details. They are deleted if you withdraw.</p>`
    : '<p>The paid programme is not open yet.</p>';
  $('#app').innerHTML = heading('My recordings', 'Your <em>voice</em>, your control.', 'Listen to or delete any recording. Deleting removes the audio for good.') +
    `<div class="stats"><div class="stat"><strong>${me.counts.accepted}</strong><span>accepted</span></div><div class="stat"><strong>${me.counts.asr_pending}</strong><span>being checked</span></div><div class="stat"><strong>${me.counts.needs_review}</strong><span>with a reviewer</span></div><div class="stat"><strong>${me.counts.rejected}</strong><span>not accepted</span></div></div>
    <div class="two-col"><section class="card"><div class="section-heading"><h3>Recordings</h3><a class="btn red small" href="#record">Record more ${arrow}</a></div>
    ${me.recordings.length ? me.recordings.map(r => `<div class="receipt-item donate-rec"><div><p lang="ne">${esc(r.text)}</p><span class="chip ${STATE[r.state][1]}">${STATE[r.state][0]}</span> <span class="micro">${r.seconds.toFixed(1)} s${r.reviewer_note ? ' · ' + esc(r.reviewer_note) : ''}</span></div><div class="review-actions"><button class="btn secondary small" data-play="${esc(r.id)}">Play</button><button class="btn danger small" data-delete="${esc(r.id)}">Delete</button></div></div>`).join('') : '<p>No recordings yet.</p>'}
    <audio id="player" hidden></audio></section>
    <aside><div class="card"><h3>Paid programme</h3>${paid}</div>
    <div class="card"><h3>Leave the programme</h3><p>Deletes every recording and your profile. Payment records are kept as financial records.</p><button class="btn danger small" id="withdraw">Withdraw and delete everything</button></div></aside></div>`;
  document.querySelectorAll('[data-play]').forEach(b => b.onclick = () => { const p = $('#player'); p.src = `/api/donate/recordings/${encodeURIComponent(b.dataset.play)}/audio`; p.play(); });
  document.querySelectorAll('[data-delete]').forEach(b => b.onclick = async () => {
    if (!confirm('Delete this recording permanently?')) return;
    try { await api(`/api/donate/recordings/${encodeURIComponent(b.dataset.delete)}`, {method: 'DELETE'}); await myVoicePage(); } catch (e) { toast(e.message); }
  });
  $('#withdraw').onclick = async () => {
    if (!confirm('Delete all your recordings and leave the programme? This cannot be undone.')) return;
    try { await api('/api/donate/withdraw', {method: 'POST'}); toast('Everything has been deleted.'); location.hash = 'donate'; } catch (e) { toast(e.message); }
  };
  const form = $('#apply');
  if (form) form.onsubmit = async (e) => {
    e.preventDefault();
    try { await api('/api/donate/paid/apply', {method: 'POST', body: JSON.stringify(Object.fromEntries(new FormData(form)))}); await myVoicePage(); } catch (err) { toast(err.message); }
  };
}

function storageCard(st) {
  if (!st || st.backend !== 'huggingface') return '<section class="card"><h3>Storage</h3><p class="muted">Local disk only. Set NEPTTS_HF_DONATIONS_BENCHMARK_REPO to keep takes on private Hugging Face datasets.</p></section>';
  const stale = st.oldest_spooled_seconds != null && st.oldest_spooled_seconds > 3600;
  const repos = Object.entries(st.repos).map(([repo, r]) => `<p><b>${esc(repo)}</b> · ${r.last_sync ? 'last sync ' + new Date(r.last_sync * 1000).toLocaleString() : 'not synced yet'}${r.last_error ? ` · <span class="error">${esc(r.last_error)}</span>` : ''}</p>`).join('');
  return `<section class="card"><h3>Storage · private Hugging Face</h3>${repos}<p class="micro">${st.spooled} take(s) waiting to upload${st.oldest_spooled_seconds != null ? `, oldest ${Math.round(st.oldest_spooled_seconds / 60)} min` : ''} · ${st.pending_deletions} deletion(s) pending</p>${stale ? '<div class="error">Takes have waited over an hour. Check the Hugging Face token and repo access.</div>' : ''}</section>`;
}

async function donationsAdmin() {
  let d;
  try { d = await api('/api/admin/donations'); }
  catch { $('#app').innerHTML = heading('Donation review', 'Reviewer sign-in needed.', '') + `<a class="btn red" href="#admin">Sign in ${arrow}</a>`; return; }
  $('#app').innerHTML = heading('Donation review', 'Donated <em>voices.</em>', `${d.donors} donors · ${d.withdrawn} withdrawn · enrollment ${d.open ? 'open' : 'closed (consent not approved)'}`) +
    storageCard(d.storage) +
    `<section class="card"><h3>Coverage</h3>${Object.entries(d.coverage).map(([set, v]) => `<p><b>${esc(set)}</b> · ${Object.entries(v).filter(([k]) => k !== 'prompts_total').map(([k, x]) => `${esc(k)} ${x.prompts}/${v.prompts_total} prompts, ${x.minutes} min`).join(' · ')}</p>`).join('') || '<p>No recordings yet.</p>'}</section>
    <section class="card"><h3>Needs review (${d.review_queue.length})</h3>${d.review_queue.map(r => `<div class="review-row"><div class="grow"><p lang="ne">${esc(r.text)}</p><span class="micro">CER ${(r.cer * 100).toFixed(1)}% · SNR ${r.signal.snr_db} dB · ${r.seconds.toFixed(1)} s</span><audio controls preload="none" src="/api/admin/donations/recordings/${esc(r.id)}/audio"></audio></div><div class="review-actions"><button class="btn small" data-decide="accept" data-id="${esc(r.id)}">Accept</button><button class="btn danger small" data-decide="reject" data-id="${esc(r.id)}">Reject</button></div></div>`).join('') || '<p>Nothing to review.</p>'}</section>
    <section class="card"><h3>Paid programme</h3>${d.applications.map(a => `<div class="review-row"><div class="grow"><b>${esc(a.payout?.name || '')}</b> · ${esc(a.payout?.method || '')} ${esc(a.payout?.handle || '')}<br><span class="micro">${a.paid_status} · ${a.accepted_minutes} accepted min · balance NPR ${a.balance_npr}</span></div><div class="review-actions">${a.paid_status === 'applied' ? `<button class="btn small" data-paid="approve" data-id="${esc(a.id)}">Approve at NPR ${d.donor_rate}/min</button><button class="btn danger small" data-paid="reject" data-id="${esc(a.id)}">Reject</button>` : `<button class="btn secondary small" data-payout="${esc(a.id)}">Record payout</button>`}</div></div>`).join('') || '<p>No applications.</p>'}</section>`;
  document.querySelectorAll('[data-decide]').forEach(b => b.onclick = async () => {
    const note = prompt('Reviewer note (shown to the donor)');
    if (!note) return;
    try { await api(`/api/admin/donations/recordings/${encodeURIComponent(b.dataset.id)}/decision`, {method: 'POST', body: JSON.stringify({action: b.dataset.decide, note})}); await donationsAdmin(); } catch (e) { toast(e.message); }
  });
  document.querySelectorAll('[data-paid]').forEach(b => b.onclick = async () => {
    try { await api(`/api/admin/donations/donors/${encodeURIComponent(b.dataset.id)}/paid`, {method: 'POST', body: JSON.stringify({action: b.dataset.paid})}); await donationsAdmin(); } catch (e) { toast(e.message); }
  });
  document.querySelectorAll('[data-payout]').forEach(b => b.onclick = async () => {
    const amount = Number(prompt('Amount paid (NPR)'));
    const reference = amount > 0 && prompt('Payment reference (e.g. eSewa transaction ID)');
    if (!reference) return;
    try { await api(`/api/admin/donations/donors/${encodeURIComponent(b.dataset.payout)}/payouts`, {method: 'POST', body: JSON.stringify({amount_npr: amount, reference})}); await donationsAdmin(); } catch (e) { toast(e.message); }
  });
}

Object.assign(window, {donatePage, recordPage, myVoicePage, donationsAdmin});
})();
