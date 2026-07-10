(() => {
  'use strict';
  const body = document.body;
  const els = {
    setup: document.getElementById('setup'), guidance: document.getElementById('guidance'), summary: document.getElementById('summary'),
    startBtn: document.getElementById('start-btn'), stopBtn: document.getElementById('stop-btn'),
    newBtn: document.getElementById('new-session-btn'), clearBtn: document.getElementById('clear-btn'),
    toggleVoice: document.getElementById('toggle-voice'), toggleHaptic: document.getElementById('toggle-haptics'), toggleListen: document.getElementById('toggle-listen'),
    voiceHint: document.getElementById('voice-hint'), loader: document.getElementById('loader'), loaderText: document.getElementById('loader-text'),
    video: document.getElementById('video'), overlay: document.getElementById('overlay'),
    pathStatus: document.getElementById('path-status'), pathStatusText: document.getElementById('path-status-text'),
    micIndicator: document.getElementById('mic-indicator'), alertCard: document.getElementById('alert-card'), alertText: document.getElementById('alert-text'),
    summaryList: document.getElementById('summary-list'), summaryEmpty: document.getElementById('summary-empty'), summaryCount: document.getElementById('summary-count'),
    liveRegion: document.getElementById('live-region'), statusRegion: document.getElementById('status-region')
  };
  const ctx = els.overlay.getContext('2d');
  const settings = { voice: true, haptics: true, listen: false };
  const state = { running: false, model: null, stream: null, rafId: null, lastDetectAt: 0, lastSpeakAt: 0, introSpoken: false, announced: new Map(), recognizer: null, recognizerWanted: false };
  const DETECT_INTERVAL = 130, SPEAK_COOLDOWN = 1400, CONFIDENCE_MIN = 0.5;
  const TIER_ORDER = { far: 0, medium: 1, close: 2 };
  const RELEVANT = new Set(['person','bicycle','car','motorcycle','bus','truck','train','chair','bench','couch','potted plant','dining table','bed','backpack','handbag','suitcase','dog','cat','traffic light','stop sign','fire hydrant','parking meter','umbrella','bottle','cup','tv','refrigerator','toilet','sink','door']);

  (function loadIcons() {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/lucide@0.462.0/dist/umd/lucide.min.js';
    s.async = true; s.onload = () => { try { window.lucide.createIcons(); } catch (e) {} };
    document.head.appendChild(s);
  })();

  function speak(text, { urgent = false } = {}) {
    els.liveRegion.textContent = text;
    if (!settings.voice || !('speechSynthesis' in window)) return;
    if (urgent) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = urgent ? 1.08 : 0.98; u.pitch = 1.0; u.volume = 1.0;
    window.speechSynthesis.speak(u);
  }
  function speakIntro() { if (state.introSpoken) return; state.introSpoken = true; speak("EchoPath is ready. Tap start guiding, then hold your phone up as you walk."); }
  function vibrate(tier) {
    if (!settings.haptics || !('vibrate' in navigator)) return;
    if (tier === 'medium') navigator.vibrate(110);
    else if (tier === 'close') navigator.vibrate([200, 80, 200]);
  }
  function showScreen(name) {
    body.dataset.screen = name;
    els.setup.hidden = name !== 'setup'; els.guidance.hidden = name !== 'guidance'; els.summary.hidden = name !== 'summary';
  }
  function announceStatus(text) { els.statusRegion.textContent = text; }
  function bindToggle(el, key, onChange) {
    el.addEventListener('click', () => { settings[key] = !settings[key]; el.setAttribute('aria-checked', String(settings[key])); if (onChange) onChange(settings[key]); });
  }
  bindToggle(els.toggleVoice, 'voice');
  bindToggle(els.toggleHaptic, 'haptics');
  bindToggle(els.toggleListen, 'listen', (on) => { els.voiceHint.hidden = !on; if (on) startRecognition(); else stopRecognition(); });

  async function startCamera() {
    const constraints = { audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } };
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia(constraints); }
    catch (e) { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true }); }
    state.stream = stream; els.video.srcObject = stream;
    await new Promise((resolve) => { if (els.video.readyState >= 2) return resolve(); els.video.onloadedmetadata = () => resolve(); });
    await els.video.play();
  }
  function stopCamera() {
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    els.video.srcObject = null;
  }
  async function loadModel() {
    if (state.model) return state.model;
    els.loader.hidden = false; els.loaderText.textContent = 'Warming up the detector…';
    state.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    return state.model;
  }
  function fitCanvas() { els.overlay.width = els.overlay.clientWidth; els.overlay.height = els.overlay.clientHeight; }
  function coverTransform() {
    const vw = els.video.videoWidth || 1, vh = els.video.videoHeight || 1, cw = els.overlay.width, ch = els.overlay.height;
    const scale = Math.max(cw / vw, ch / vh);
    return { scale, offsetX: (cw - vw * scale) / 2, offsetY: (ch - vh * scale) / 2, vw, vh };
  }
  function tierFor(bbox, vw, vh) {
    const [, , w, h] = bbox; const areaRatio = (w * h) / (vw * vh); const heightRatio = h / vh;
    if (areaRatio > 0.17 || heightRatio > 0.72) return 'close';
    if (areaRatio > 0.055 || heightRatio > 0.42) return 'medium';
    return 'far';
  }
  function positionFor(bbox, vw) {
    const [x, , w] = bbox; const centerX = (x + w / 2) / vw;
    if (centerX < 0.38) return 'left'; if (centerX > 0.62) return 'right'; return 'center';
  }
  const wallCanvas = document.createElement('canvas'); wallCanvas.width = 40; wallCanvas.height = 40;
  const wallCtx = wallCanvas.getContext('2d', { willReadFrequently: true });
  function detectWall() {
    const vw = els.video.videoWidth, vh = els.video.videoHeight; if (!vw || !vh) return false;
    const cropW = vw * 0.6, cropH = vh * 0.6, sx = (vw - cropW) / 2, sy = (vh - cropH) / 2;
    wallCtx.drawImage(els.video, sx, sy, cropW, cropH, 0, 0, 40, 40);
    const { data } = wallCtx.getImageData(0, 0, 40, 40);
    let sum = 0, sumSq = 0; const n = 40 * 40;
    for (let i = 0; i < data.length; i += 4) { const lum = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]; sum += lum; sumSq += lum * lum; }
    const mean = sum / n; const stdDev = Math.sqrt(sumSq / n - mean * mean);
    return stdDev < 12 && mean > 40;
  }
  const TIER_COLOR = { far: 'oklch(80% 0.14 158)', medium: 'oklch(83% 0.14 92)', close: 'oklch(66% 0.2 26)' };
  function drawDetections(dets, t) {
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    ctx.lineWidth = 3; ctx.font = '600 15px "IBM Plex Mono", monospace'; ctx.textBaseline = 'top';
    dets.forEach((d) => {
      const [x, y, w, h] = d.bbox;
      const dx = x * t.scale + t.offsetX, dy = y * t.scale + t.offsetY, dw = w * t.scale, dh = h * t.scale;
      const color = TIER_COLOR[d.tier];
      ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = d.tier === 'close' ? 18 : 8;
      roundRect(dx, dy, dw, dh, 10); ctx.stroke(); ctx.shadowBlur = 0;
      const label = d.class + ' · ' + d.tier; const tw = ctx.measureText(label).width + 16;
      ctx.fillStyle = 'oklch(10% 0.03 255 / 0.82)'; roundRect(dx, Math.max(0, dy - 26), tw, 22, 6); ctx.fill();
      ctx.fillStyle = color; ctx.fillText(label, dx + 8, Math.max(2, dy - 23));
    });
  }
  function roundRect(x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function phrase(label, tier, position) {
    const name = label === 'wall' ? 'Wall' : capitalize(label);
    if (tier === 'close') {
      const dodge = position === 'center' ? 'stop' : ('ease ' + (position === 'left' ? 'right' : 'left'));
      return name + ', close, ' + position + '. ' + capitalize(dodge) + '.';
    }
    if (tier === 'medium') return name + ' ' + (position === 'center' ? 'ahead' : 'to your ' + position) + ', coming up.';
    return name + ' ' + (position === 'center' ? 'ahead' : 'to your ' + position) + '.';
  }
  function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function setPathStatus(tier) {
    const map = { clear: ['path-status--clear', "You're clear to walk"], medium: ['path-status--caution', 'Something coming up'], close: ['path-status--urgent', 'Obstacle close, take care'] };
    const key = tier === 'close' ? 'close' : tier === 'medium' ? 'medium' : 'clear';
    const [cls, text] = map[key]; els.pathStatus.className = 'path-status ' + cls; els.pathStatusText.textContent = text;
  }
  let alertTimer = null;
  function showAlert(text, tier) {
    els.alertText.textContent = text; els.alertCard.dataset.tier = tier; els.alertCard.hidden = false;
    els.alertCard.style.animation = 'none'; void els.alertCard.offsetHeight; els.alertCard.style.animation = '';
    clearTimeout(alertTimer); alertTimer = setTimeout(() => { els.alertCard.hidden = true; }, 3200);
  }
  async function logEvent(label, tier, position) {
    try { await fetch('/api/log', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label, distance: tier, position, timestamp: new Date().toISOString() }) }); } catch (e) {}
  }
  function shouldAnnounce(key, tier) { const prev = state.announced.get(key); if (prev === undefined) return true; return TIER_ORDER[tier] > TIER_ORDER[prev]; }
  function handleAlerts(items) {
    const now = performance.now();
    items.sort((a, b) => TIER_ORDER[b.tier] - TIER_ORDER[a.tier]);
    setPathStatus(items.length ? items[0].tier : 'clear');
    const seen = new Set();
    for (const it of items) {
      seen.add(it.key);
      if (it.tier === 'far') { state.announced.set(it.key, 'far'); continue; }
      const urgent = it.tier === 'close';
      const canSpeak = urgent || (now - state.lastSpeakAt > SPEAK_COOLDOWN);
      if (shouldAnnounce(it.key, it.tier) && canSpeak) {
        const text = phrase(it.label, it.tier, it.position);
        speak(text, { urgent }); showAlert(text, it.tier); vibrate(it.tier); logEvent(it.label, it.tier, it.position);
        state.lastSpeakAt = now; state.announced.set(it.key, it.tier);
        if (urgent) break;
      } else { state.announced.set(it.key, it.tier); }
    }
    for (const key of state.announced.keys()) { if (!seen.has(key)) state.announced.delete(key); }
  }
  async function loop() {
    if (!state.running) return;
    const now = performance.now();
    if (now - state.lastDetectAt >= DETECT_INTERVAL) {
      state.lastDetectAt = now;
      let predictions = [];
      try { predictions = await state.model.detect(els.video, 20); } catch (e) {}
      const t = coverTransform(); const items = [];
      predictions.forEach((p) => {
        if (p.score < CONFIDENCE_MIN) return; if (!RELEVANT.has(p.class)) return;
        items.push({ class: p.class, label: p.class, tier: tierFor(p.bbox, t.vw, t.vh), position: positionFor(p.bbox, t.vw), bbox: p.bbox, key: p.class });
      });
      const hasCloseCenter = items.some((i) => i.tier === 'close' && i.position === 'center');
      if (!hasCloseCenter && detectWall()) {
        items.push({ class: 'wall', label: 'wall', tier: 'close', position: 'center', bbox: [t.vw * 0.25, t.vh * 0.2, t.vw * 0.5, t.vh * 0.6], key: 'wall' });
      }
      drawDetections(items, t); handleAlerts(items);
    }
    state.rafId = requestAnimationFrame(loop);
  }
  async function startGuiding() {
    if (state.running) return;
    try {
      showScreen('guidance'); fitCanvas(); await loadModel();
      els.loaderText.textContent = 'Turning on the camera…'; await startCamera(); fitCanvas();
      els.loader.hidden = true; state.running = true; state.announced.clear(); setPathStatus('clear');
      speak("You're all set. I'll let you know what's ahead."); announceStatus('Guidance started.'); loop();
    } catch (e) {
      els.loader.hidden = true; showScreen('setup');
      speak("I couldn't reach the camera. Please allow camera access and try again.", { urgent: true }); announceStatus('Camera unavailable.');
    }
  }
  async function stopGuiding() {
    if (!state.running) { showSummary(); return; }
    state.running = false; if (state.rafId) cancelAnimationFrame(state.rafId);
    window.speechSynthesis && window.speechSynthesis.cancel(); stopCamera();
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height); els.alertCard.hidden = true;
    speak('Guidance stopped. Here is your session summary.'); await showSummary();
  }
  async function showSummary() {
    showScreen('summary'); els.summaryList.innerHTML = '';
    let logs = [];
    try { const res = await fetch('/api/logs'); const data = await res.json(); logs = data.logs || []; } catch (e) {}
    if (!logs.length) { els.summaryEmpty.hidden = false; els.summaryCount.textContent = 'No obstacles logged this session.'; return; }
    els.summaryEmpty.hidden = true;
    els.summaryCount.textContent = logs.length + ' ' + (logs.length === 1 ? 'obstacle' : 'obstacles') + ' logged.';
    const frag = document.createDocumentFragment();
    logs.forEach((ev) => {
      const li = document.createElement('li'); li.className = 'log-item';
      const time = new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      li.innerHTML = '<span class="log-item__badge log-item__badge--' + ev.distance + '" aria-hidden="true"></span><span class="log-item__main"><span class="log-item__label">' + ev.label + '</span><span class="log-item__meta">' + ev.distance + ' · ' + ev.position + '</span></span><span class="log-item__time">' + time + '</span>';
      frag.appendChild(li);
    });
    els.summaryList.appendChild(frag); if (window.lucide) window.lucide.createIcons();
  }
  async function clearSession() {
    try { await fetch('/api/logs', { method: 'DELETE' }); } catch (e) {}
    els.summaryList.innerHTML = ''; els.summaryEmpty.hidden = false; els.summaryCount.textContent = 'Session cleared.'; speak('Session cleared.');
  }
  function getRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition; if (!SR) return null;
    const r = new SR(); r.continuous = true; r.interimResults = true; r.lang = 'en-US'; return r;
  }
  function startRecognition() {
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      speak('Voice commands are not supported in this browser.'); els.toggleListen.setAttribute('aria-checked', 'false'); settings.listen = false; els.voiceHint.hidden = true; return;
    }
    state.recognizerWanted = true; els.micIndicator.hidden = false; if (state.recognizer) return;
    const r = getRecognition(); state.recognizer = r;
    r.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) transcript += event.results[i][0].transcript;
      const heard = transcript.toLowerCase();
      if (/(echo\s?path|echo\s?pat).*(start|begin|go)/.test(heard) || /start guiding/.test(heard)) { if (!state.running) startGuiding(); }
      else if (/(echo\s?path|echo\s?pat).*(stop|end|pause)/.test(heard)) { if (state.running) stopGuiding(); }
    };
    r.onend = () => { if (state.recognizerWanted) { try { r.start(); } catch (e) {} } };
    r.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        speak('Microphone access was blocked, so voice commands are off.'); stopRecognition();
        els.toggleListen.setAttribute('aria-checked', 'false'); settings.listen = false; els.voiceHint.hidden = true;
      }
    };
    try { r.start(); } catch (e) {}
  }
  function stopRecognition() {
    state.recognizerWanted = false; els.micIndicator.hidden = true;
    if (state.recognizer) { try { state.recognizer.stop(); } catch (e) {} state.recognizer = null; }
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden && state.recognizerWanted && !state.recognizer) startRecognition(); });
  els.startBtn.addEventListener('click', startGuiding);
  els.stopBtn.addEventListener('click', stopGuiding);
  els.newBtn.addEventListener('click', startGuiding);
  els.clearBtn.addEventListener('click', clearSession);
  window.addEventListener('resize', () => { if (state.running) fitCanvas(); });
  window.addEventListener('load', () => { setTimeout(speakIntro, 400); window.lucide && window.lucide.createIcons(); });
  window.addEventListener('pointerdown', function once() { speakIntro(); window.removeEventListener('pointerdown', once); }, { once: true });
})();
