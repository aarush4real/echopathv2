const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR = path.join(__dirname, 'data');
const LOGS_FILE = path.join(DATA_DIR, 'logs.json');

function ensureStore() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  if (!fs.existsSync(LOGS_FILE)) fs.writeFileSync(LOGS_FILE, '[]', 'utf8');
}
function readLogs() {
  try {
    const parsed = JSON.parse(fs.readFileSync(LOGS_FILE, 'utf8'));
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) { return []; }
}
function writeLogs(logs) {
  fs.writeFileSync(LOGS_FILE, JSON.stringify(logs, null, 2), 'utf8');
}

app.use(express.json({ limit: '256kb' }));
app.use(express.static(PUBLIC_DIR));

app.get('/api/health', (req, res) => {
  res.json({ ok: true, service: 'echopath', time: new Date().toISOString() });
});

app.post('/api/log', (req, res) => {
  const { label, distance, position } = req.body || {};
  if (!label || !distance || !position) {
    return res.status(400).json({ ok: false, error: 'label, distance and position are required.' });
  }
  const event = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
    label: String(label), distance: String(distance), position: String(position),
    timestamp: req.body.timestamp || new Date().toISOString()
  };
  const logs = readLogs();
  logs.push(event);
  writeLogs(logs);
  res.status(201).json({ ok: true, event });
});

app.get('/api/logs', (req, res) => {
  const logs = readLogs();
  res.json({ ok: true, count: logs.length, logs: logs.slice().reverse() });
});

app.delete('/api/logs', (req, res) => {
  writeLogs([]);
  res.json({ ok: true, cleared: true });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

ensureStore();
app.listen(PORT, () => {
  console.log('');
  console.log('  EchoPath v2 is running.');
  console.log('  Open  http://localhost:' + PORT);
  console.log('  Logs stored in  ' + LOGS_FILE);
  console.log('');
});
