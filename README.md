# EchoPath v2

An AR blind-assist navigator. Point a phone or laptop camera forward while walking, and EchoPath detects obstacles in real time and warns you with calm spoken guidance and vibration, like AI-powered echolocation. Every warning is logged so a caregiver can review the walk afterward.

## Setup
Requires Node.js 16+.
    npm install
    npm start
Then open http://localhost:3000 in Chrome.

## Browser permissions
- Camera: required.
- Microphone: only if you enable the optional "Voice commands" toggle for hands-free "EchoPath start / stop". Core navigation does not need it.

## Known limitations
- Distance is estimated from bounding-box size, not true depth (no LiDAR on a single camera).
- Wall detection is a visual-uniformity heuristic, not a real depth reading.
- Object vocabulary is the fixed 80 COCO classes. Always use alongside a cane or guide dog, never instead of one.

## Future ideas
- True depth on LiDAR phones, offline model caching, multilingual voice, custom training for stairs and curbs.

## API
- GET  /api/health   health check
- POST /api/log      save { label, distance, position, timestamp }
- GET  /api/logs     full session log (newest first)
- DELETE /api/logs   clear the log

## Push to GitHub
Create an empty repo at https://github.com/new then:
    git init
    git add .
    git commit -m "Initial commit: EchoPath v2"
    git branch -M main
    git remote add origin https://github.com/your-name/echopath.git
    git push -u origin main

## License
MIT.
