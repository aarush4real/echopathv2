/* ===========================================================================
   EchoPath v2 â€” client logic
   Runs entirely in the browser. No backend, no session summary.
     â€¢ getUserMedia camera + TensorFlow.js / COCO-SSD detection
     â€¢ Wall-ahead heuristic (visual uniformity)
     â€¢ Speech Synthesis for calm spoken warnings
     â€¢ Vibration API for haptics
     â€¢ Speech Recognition: "EchoPath start" opens the camera, "EchoPath stop" ends it
   =========================================================================== */

(() => {
  'use strict';

  const body = document.body;
  const els = {
    setup:        document.getElementById('setup'),
    guidance:     document.getElementById('guidance'),
    startBtn:     document.getElementById('start-btn'),
    stopBtn:      document.getElementById('stop-btn'),
    toggleVoice:  document.getElementById('toggle-voice'),
    toggleHaptic: document.getElementById('toggle-haptics'),
    toggleListen: document.getElementById('toggle-listen'),
    voiceHint:    document.getElementById('voice-hint'),
    loader:       document.getElementById('loader'),
    loaderText:   document.getElementById('loader-text'),
    video:        document.getElementById('video'),
    overlay:      document.getElementById('overlay'),
    pathStatus:   document.getElementById('path-status'),
    pathStatusText: document.getElementById('path-status-text'),
    micIndicator: document.getElementById('mic-indicator'),
    alertCard:    document.getElementById('alert-card'),
    alertText:    document.getElementById('alert-text'),
    liveRegion:   document.getElementById('live-region'),
    statusRegion: document.getElementById('status-region')
  };
  const ctx = els.overlay.getContext('2d');

  // Voice commands default ON so the app listens the moment you allow the mic.
  const settings = { voice: true, haptics: true, listen: true };

  const state = {
    running: false, model: null, stream: null, rafId: null,
    lastDetectAt: 0, lastSpeakAt: 0, introSpoken: false,
    announced: new Map(),
    recognizer: null, recognizerWanted: false, recogRunning: false,
    // Wall needs to persist a couple frames before we call it, to cut false alarms.
    wallStreak: 0
  };

  const DETECT_INTERVAL = 130;
  const SPEAK_COOLDOWN  = 1400;
  const CONFIDENCE_MIN  = 0.5;
  const TIER_ORDER = { far: 0, medium: 1, close: 2 };

  const RELEVANT = new Set([
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'train',
    'chair', 'bench', 'couch', 'potted plant', 'dining table', 'bed',
    'backpack', 'handbag', 'suitcase', 'dog', 'cat', 'traffic light',
    'stop sign', 'fire hydrant', 'parking meter', 'umbrella', 'bottle',
    'cup', 'tv', 'refrigerator', 'toilet', 'sink', 'door'
  ]);

  (function loadIcons() {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/lucide@0.462.0/dist/umd/lucide.min.js';
    s.async = true;
    s.onload = () => { try { window.lucide.createIcons(); } catch (e) {} };
    document.head.appendChild(s);
  })();

  /* ---------- Speech output ---------- */
  function speak(text, { urgent = false } = {}) {
    els.liveRegion.textContent = text;
    if (!settings.voice || !('speechSynthesis' in window)) return;
    if (urgent) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = urgent ? 1.08 : 0.98; u.pitch = 1.0; u.volume = 1.0;
    window.speechSynthesis.speak(u);
  }
  function speakIntro() {
    if (state.introSpoken) return;
    state.introSpoken = true;
    speak("EchoPath is ready. Say EchoPath start, or tap the button, to begin.");
  }

  /* ---------- Haptics ---------- */
  function vibrate(tier) {
    if (!settings.haptics || !('vibrate' in navigator)) return;
    if (tier === 'medium') navigator.vibrate(110);
    else if (tier === 'close') navigator.vibrate([200, 80, 200]);
  }

  /* ---------- Screens ---------- */
  function showScreen(name) {
    body.dataset.screen = name;
    els.setup.hidden = name !== 'setup';
    els.guidance.hidden = name !== 'guidance';
  }
  function announceStatus(text) { els.statusRegion.textContent = text; }

  /* ---------- Toggles ---------- */
  function bindToggle(el, key, onChange) {
    el.addEventListener('click', () => {
      settings[key] = !settings[key];
      el.setAttribute('aria-checked', String(settings[key]));
      if (onChange) onChange(settings[key]);
    });
  }
  bindToggle(els.toggleVoice, 'voice');
  bindToggle(els.toggleHaptic, 'haptics');
  bindToggle(els.toggleListen, 'listen', (on) => {
    els.voiceHint.hidden = !on;
    if (on) startRecognition(); else stopRecognition();
  });

  /* ---------- Camera ---------- */
  async function startCamera() {
    const constraints = {
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }
    };
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia(constraints); }
    catch (e) { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true }); }
    state.stream = stream;
    els.video.srcObject = stream;
    await new Promise((resolve) => {
      if (els.video.readyState >= 2) return resolve();
      els.video.onloadedmetadata = () => resolve();
    });
    await els.video.play();
  }
  function stopCamera() {
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    els.video.srcObject = null;
  }

  /* ---------- Model ---------- */
  async function loadModel() {
    if (state.model) return state.model;
    els.loader.hidden = false;
    els.loaderText.textContent = 'Warming up the detectorâ€¦';
    state.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    return state.model;
  }

  /* ---------- Geometry ---------- */
  function fitCanvas() { els.overlay.width = els.overlay.clientWidth; els.overlay.height = els.overlay.clientHeight; }
  function coverTransform() {
    const vw = els.video.videoWidth || 1, vh = els.video.videoHeight || 1;
    const cw = els.overlay.width, ch = els.overlay.height;
    const scale = Math.max(cw / vw, ch / vh);
    return { scale, offsetX: (cw - vw * scale) / 2, offsetY: (ch - vh * scale) / 2, vw, vh };
  }
  function tierFor(bbox, vw, vh) {
    const [, , w, h] = bbox;
    const areaRatio = (w * h) / (vw * vh), heightRatio = h / vh;
    if (areaRatio > 0.17 || heightRatio > 0.72) return 'close';
    if (areaRatio > 0.055 || heightRatio > 0.42) return 'medium';
    return 'far';
  }
  function positionFor(bbox, vw) {
    const [x, , w] = bbox; const centerX = (x + w / 2) / vw;
    if (centerX < 0.38) return 'left';
    if (centerX > 0.62) return 'right';
    return 'center';
  }

  /* ---------- Wall heuristic (loosened + streak-confirmed) ----------
     A flat surface up close fills the center with a large, low-detail,
     uniform region. We downsample the central area and measure luminance
     spread. Low spread + reasonable brightness = a wall filling the view.
     We require it to hold for a few consecutive frames to avoid flicker. */
  const wallCanvas = document.createElement('canvas');
  wallCanvas.width = 48; wallCanvas.height = 48;
  const wallCtx = wallCanvas.getContext('2d', { willReadFrequently: true });

  function wallLikely() {
    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    if (!vw || !vh) return false;
    const cropW = vw * 0.7, cropH = vh * 0.7;
    const sx = (vw - cropW) / 2, sy = (vh - cropH) / 2;
    wallCtx.drawImage(els.video, sx, sy, cropW, cropH, 0, 0, 48, 48);
    const { data } = wallCtx.getImageData(0, 0, 48, 48);
    let sum = 0, sumSq = 0; const n = 48 * 48;
    for (let i = 0; i < data.length; i += 4) {
      const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      sum += lum; sumSq += lum * lum;
    }
    const mean = sum / n;
    const stdDev = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
    // Loosened: most indoor/outdoor walls read stdDev ~10-28 up close.
    return stdDev < 26 && mean > 28 && mean < 245;
  }

  /* ---------- Drawing ---------- */
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
      const label = d.class + ' Â· ' + d.tier; const tw = ctx.measureText(label).width + 16;
      ctx.fillStyle = 'oklch(10% 0.03 255 / 0.82)'; roundRect(dx, Math.max(0, dy - 26), tw, 22, 6); ctx.fill();
      ctx.fillStyle = color; ctx.fillText(label, dx + 8, Math.max(2, dy - 23));
    });
  }
  function roundRect(x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  /* ---------- Phrasing ---------- */
  function phrase(label, tier, position) {
    const name = label === 'wall' ? 'Wall' : capitalize(label);
    if (label === 'wall') return 'Wall ahead, close. Stop.';
    if (tier === 'close') {
      const dodge = position === 'center' ? 'stop' : ('ease ' + (position === 'left' ? 'right' : 'left'));
      return name + ', close, ' + position + '. ' + capitalize(dodge) + '.';
    }
    if (tier === 'medium') return name + ' ' + (position === 'center' ? 'ahead' : 'to your ' + position) + ', coming up.';
    return name + ' ' + (position === 'center' ? 'ahead' : 'to your ' + position) + '.';
  }
  function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  /* ---------- Status pill ---------- */
  function setPathStatus(tier) {
    const map = {
      clear:  ['path-status--clear', "You're clear to walk"],
      medium: ['path-status--caution', 'Something coming up'],
      close:  ['path-status--urgent', 'Obstacle close, take care']
    };
    const key = tier === 'close' ? 'close' : tier === 'medium' ? 'medium' : 'clear';
    const [cls, text] = map[key];
    els.pathStatus.className = 'path-status ' + cls;
    els.pathStatusText.textContent = text;
  }

  /* ---------- Alert card ---------- */
  let alertTimer = null;
  function showAlert(text, tier) {
    els.alertText.textContent = text; els.alertCard.dataset.tier = tier; els.alertCard.hidden = false;
    els.alertCard.style.animation = 'none'; void els.alertCard.offsetHeight; els.alertCard.style.animation = '';
    clearTimeout(alertTimer); alertTimer = setTimeout(() => { els.alertCard.hidden = true; }, 3200);
  }

  /* ---------- Announce decision ---------- */
  function shouldAnnounce(key, tier) {
    const prev = state.announced.get(key);
    if (prev === undefined) return true;
    return TIER_ORDER[tier] > TIER_ORDER[prev];
  }
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
        speak(text, { urgent }); showAlert(text, it.tier); vibrate(it.tier);
        state.lastSpeakAt = now; state.announced.set(it.key, it.tier);
        if (urgent) break;
      } else { state.announced.set(it.key, it.tier); }
    }
    for (const key of state.announced.keys()) if (!seen.has(key)) state.announced.delete(key);
  }

  /* ---------- Detection loop ---------- */
  async function loop() {
    if (!state.running) return;
    const now = performance.now();
    if (now - state.lastDetectAt >= DETECT_INTERVAL) {
      state.lastDetectAt = now;
      let predictions = [];
      try { predictions = await state.model.detect(els.video, 20); } catch (e) {}
      const t = coverTransform();
      const items = [];
      predictions.forEach((p) => {
        if (p.score < CONFIDENCE_MIN || !RELEVANT.has(p.class)) return;
        items.push({ class: p.class, label: p.class, tier: tierFor(p.bbox, t.vw, t.vh), position: positionFor(p.bbox, t.vw), bbox: p.bbox, key: p.class });
      });

      // Wall check: skip if a real close object already explains the view.
      const hasCloseCenter = items.some((i) => i.tier === 'close' && i.position === 'center');
      if (!hasCloseCenter && wallLikely()) {
        state.wallStreak++;
      } else {
        state.wallStreak = 0;
      }
      if (state.wallStreak >= 3) {
        items.push({ class: 'wall', label: 'wall', tier: 'close', position: 'center', bbox: [t.vw * 0.2, t.vh * 0.15, t.vw * 0.6, t.vh * 0.7], key: 'wall' });
      }

      drawDetections(items, t);
      handleAlerts(items);
    }
    state.rafId = requestAnimationFrame(loop);
  }

  /* ---------- Start / stop guiding ---------- */
  async function startGuiding() {
    if (state.running) return;
    try {
      showScreen('guidance'); fitCanvas(); await loadModel();
      els.loaderText.textContent = 'Turning on the cameraâ€¦';
      await startCamera(); fitCanvas();
      els.loader.hidden = true;
      state.running = true; state.announced.clear(); state.wallStreak = 0;
      setPathStatus('clear');
      speak("You're all set. I'll let you know what's ahead.");
      announceStatus('Guidance started.');
      loop();
    } catch (e) {
      els.loader.hidden = true; showScreen('setup');
      speak("I couldn't reach the camera. Please allow camera access and try again.", { urgent: true });
      announceStatus('Camera unavailable.');
    }
  }
  function stopGuiding() {
    if (!state.running) { showScreen('setup'); return; }
    state.running = false;
    if (state.rafId) cancelAnimationFrame(state.rafId);
    window.speechSynthesis && window.speechSynthesis.cancel();
    stopCamera();
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    els.alertCard.hidden = true;
    showScreen('setup');
    speak('Guidance stopped.');
    announceStatus('Guidance stopped.');
  }

  /* ---------- Voice commands ----------
     "EchoPath start" -> opens camera / guidance.
     "EchoPath stop"  -> stops and returns to the start screen. */
  function speechSupported() {
    return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
  }
  function getRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;
    const r = new SR();
    r.continuous = true; r.interimResults = true; r.lang = 'en-US';
    return r;
  }
  function handleTranscript(raw) {
    // Normalize: strip spaces so "echo path", "echopath", "eco path" all match.
    const heard = raw.toLowerCase();
    const squished = heard.replace(/\s+/g, '');
    const hasWake = /(echo?pa?th|ecopath|echopath|echopat)/.test(squished);
    if (!hasWake) return;
    if (/(stop|end|pause|halt)/.test(heard)) {
      if (state.running) stopGuiding();
    } else if (/(start|begin|go|open|guide)/.test(heard)) {
      if (!state.running) startGuiding();
    }
  }
  function startRecognition() {
    if (!speechSupported()) {
      speak('Voice commands are not supported in this browser. Try Chrome.');
      els.toggleListen.setAttribute('aria-checked', 'false');
      settings.listen = false; els.voiceHint.hidden = true;
      return;
    }
    state.recognizerWanted = true;
    els.micIndicator.hidden = false;
    if (state.recognizer && state.recogRunning) return;

    const r = getRecognition();
    state.recognizer = r;

    r.onstart = () => { state.recogRunning = true; };
    r.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        handleTranscript(event.results[i][0].transcript);
      }
    };
    r.onend = () => {
      state.recogRunning = false;
      // Chrome auto-stops periodically; restart while the user still wants it.
      if (state.recognizerWanted) { try { r.start(); } catch (e) {} }
    };
    r.onerror = (e) => {
      state.recogRunning = false;
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        speak('Microphone access was blocked, so voice commands are off. Allow the mic to use them.');
        stopRecognition();
        els.toggleListen.setAttribute('aria-checked', 'false');
        settings.listen = false; els.voiceHint.hidden = true;
      }
      // 'no-speech' / 'aborted' just let onend restart it.
    };
    try { r.start(); } catch (e) { /* already starting */ }
  }
  function stopRecognition() {
    state.recognizerWanted = false;
    els.micIndicator.hidden = true;
    if (state.recognizer) { try { state.recognizer.stop(); } catch (e) {} state.recognizer = null; }
    state.recogRunning = false;
  }

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && state.recognizerWanted && !state.recogRunning) startRecognition();
  });

  /* ---------- Events ---------- */
  els.startBtn.addEventListener('click', startGuiding);
  els.stopBtn.addEventListener('click', stopGuiding);
  window.addEventListener('resize', () => { if (state.running) fitCanvas(); });

  // Speech synth + recognition both need a user gesture to unlock. On the first
  // interaction we speak the intro AND kick off listening if the toggle is on.
  window.addEventListener('load', () => {
    setTimeout(speakIntro, 400);
    window.lucide && window.lucide.createIcons();
  });
  window.addEventListener('pointerdown', function once() {
    speakIntro();
    if (settings.listen) startRecognition();
    window.removeEventListener('pointerdown', once);
  }, { once: true });
})();
