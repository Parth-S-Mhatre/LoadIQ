# ⚡ LoadIQ — energy-analytics (Frontend)

<div align="center">

![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase%20Hosting-Deployed-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-WebGL%20Globe-black?style=for-the-badge&logo=threedotjs&logoColor=white)
[![Live](https://img.shields.io/badge/🌐%20Live%20App-loadiq--smart--ai.web.app-22c55e?style=for-the-badge)](https://loadiq-smart-ai.web.app/)

React + TypeScript frontend for the LoadIQ smart energy forecasting platform.
Demand-based loading · Skeleton-first UI · AI chatbot · WebGL globe

</div>

---

## 🌐 Live Application

**[https://loadiq-smart-ai.web.app/](https://loadiq-smart-ai.web.app/)**

Deployed on Firebase Hosting. Always available. For live predictions, the backend servers must be running (see [Backend README](../Backend/README.md)).

---

## 📁 Project Structure

```
energy-analytics/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard.js          # Main dashboard with load charts
│   │   ├── EnergyChatbot.js      # AI chatbot UI component
│   │   ├── Globe.js              # Three.js WebGL interactive globe
│   │   └── SkeletonLoader.js     # Skeleton screen components
│   │
│   ├── services/
│   │   ├── api.js                # Unified API layer, fetch helpers, WebGL recovery
│   │   ├── ChatbotService.js     # Gemini 2.0 Flash integration
│   │   └── AnalyticsService.js   # Telemetry, batch predictions, overview data
│   │
│   ├── pages/                    # Route-level page components (lazy loaded)
│   │   ├── HomePage.js
│   │   ├── AnalyticsPage.js
│   │   ├── ForecastPage.js
│   │   └── InsightsPage.js
│   │
│   ├── index.tsx                 # App entry point
│   └── App.tsx                   # Router + global layout
│
├── .env                          # Environment variables (never commit)
├── .env.example                  # Template for environment setup
├── package.json
└── README.md                     # This file
```

---

## ⚙️ Setup & Development

### Prerequisites

| Tool | Version |
|------|---------|
| Node.js | 18+ |
| npm | 9+ |

### 1. Install dependencies

```bash
cd energy-analytics
npm install
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Gemini AI chatbot
REACT_APP_GEMINI_API_KEY=your_gemini_api_key_here

# Backend URLs (update if deploying backend to cloud)
REACT_APP_MODEL1_URL=http://127.0.0.1:8001
REACT_APP_MODEL2_URL=http://127.0.0.1:8002
```

> ⚠️ Never commit `.env` to Git. Your API key must stay private.

### 3. Start development server

```bash
npm start
```

Opens [http://localhost:3000](http://localhost:3000)

### 4. Build for production

```bash
npm run build
```

Outputs to `build/` directory.

---

## 🚀 Deploy to Firebase

### First-time setup

```bash
npm install -g firebase-tools
firebase login
firebase init hosting
```

Select `build` as your public directory and configure as a single-page app (`Yes` to rewrite all URLs to `index.html`).

### Deploy

```bash
npm run build
firebase deploy --only hosting
```

**Live at:** [https://loadiq-smart-ai.web.app/](https://loadiq-smart-ai.web.app/)

---

## 🧠 Key Services

### `api.js` — Unified API Layer

Handles all communication with the backend. Key features:

**Connection-refused guard**
Before every API call, `isServerReachable()` pings `/health`. If the backend is offline, the API returns structured mock data instead of crashing the dashboard.

```js
import { API } from './services/api';

// Single prediction
const result = await API.predictLoad({ hour: 10, day_of_week: 1, month: 1 });

// Batch forecast
const batch = await API.getBatchPredictions({ last_24_hours: [...], horizon: 24 });

// Health check
const status = await API.healthCheck();
// { model1: { status: "ok" }, model2: { status: "offline" } }
```

**WebGL context loss recovery**
Three.js loses its WebGL context when the tab is backgrounded on mobile or the GPU process is killed. Use `initWebGLRecovery` to prevent a black, frozen canvas:

```js
import { initWebGLRecovery } from './services/api';

useEffect(() => {
  const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current });
  initWebGLRecovery(renderer, () => {
    renderer.setSize(width, height);
    renderer.render(scene, camera);
  });
  return () => renderer.dispose();
}, []);
```

---

### `ChatbotService.js` — Gemini AI Chatbot

Connects to Google Gemini 2.0 Flash. Country-aware energy domain expert.

```js
import { ChatbotService } from './services/ChatbotService';

const response = await ChatbotService.sendMessage(
  "What is the evening peak pattern in the UK?",
  "UK"
);
```

**Country profiles trained into the system prompt:**
- 🇩🇪 **Germany** — renewables-heavy, midday solar peaks, north-south transmission bottlenecks
- 🇬🇧 **UK** — offshore wind dominant, tea-time evening spike, interconnectors to France/Norway
- 🇺🇸 **USA** — fragmented grids (ERCOT/PJM/CAISO), summer cooling peaks, Duck Curve
- 🇮🇳 **India** — fast-growing demand, evening peaks, coal-heavy with rapid solar scaling

**Graceful fallback:** If Gemini API is unreachable, the chatbot returns locally-stored country pattern data instead of showing an error.

---

### `AnalyticsService.js` — Data & Telemetry

```js
import { AnalyticsService } from './services/AnalyticsService';

// Overview dashboard data
const telemetry = await AnalyticsService.getOverviewTelemetry();

// Batch prediction for analytics page
const forecast = await AnalyticsService.getBatchPredictions({
  last_24_hours: recentReadings,
  horizon: 48,
});
```

---

## 🎨 Smart UI Patterns

### Demand-based page loading

All route-level pages are lazy-loaded using React's `React.lazy()` and `Suspense`. A page's JavaScript bundle is only downloaded when the user navigates to it — not on initial app load.

```js
const AnalyticsPage = React.lazy(() => import('./pages/AnalyticsPage'));
const ForecastPage  = React.lazy(() => import('./pages/ForecastPage'));

<Suspense fallback={<SkeletonLoader />}>
  <Routes>
    <Route path="/analytics" element={<AnalyticsPage />} />
    <Route path="/forecast"  element={<ForecastPage />} />
  </Routes>
</Suspense>
```

### Skeleton-first UI

Every backend-dependent page renders a skeleton screen immediately on mount, then replaces it with real data when the API responds. The interface never shows a blank page.

```js
const [loading, setLoading] = useState(true);
const [data, setData]       = useState(null);

useEffect(() => {
  API.predictLoad(payload).then(result => {
    setData(result);
    setLoading(false);
  });
}, []);

return loading ? <SkeletonLoader /> : <Dashboard data={data} />;
```

---

## 🌐 Three.js Globe

The interactive WebGL globe visualises the four countries LoadIQ covers. Built with Three.js, it shows glowing grid nodes over each country with hover-state labels and animated connection lines.

**Known issue resolved:** WebGL context loss (GPU throttling) is now handled by `initWebGLRecovery()` in `api.js`. The globe rebuilds automatically after context restoration.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `react` `react-dom` | Core UI framework |
| `typescript` | Type safety across the codebase |
| `three` | WebGL interactive globe |
| `recharts` | Load forecast charts and analytics |
| `axios` | HTTP client (legacy, replaced with fetch in chatbot) |
| `react-router-dom` | Client-side routing with lazy loading |
| `firebase` | Hosting + optional Firestore logging |

---

## 🔒 Environment Variable Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `REACT_APP_GEMINI_API_KEY` | Yes | Google Gemini API key |
| `REACT_APP_MODEL1_URL` | Optional | Backend Model1 base URL (default: `http://127.0.0.1:8001`) |
| `REACT_APP_MODEL2_URL` | Optional | Backend Model2 base URL (default: `http://127.0.0.1:8002`) |

---

## 🐛 Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ERR_CONNECTION_REFUSED` on port 8001/8002 | Backend not running | Start `python Model1.py` and `python Model2.py` |
| `Gemini 404` | Deprecated model name | Ensure `ChatbotService.js` uses `gemini-2.0-flash` |
| `THREE.WebGLRenderer: Context Lost` | GPU process killed / tab backgrounded | Call `initWebGLRecovery(renderer, rebuildFn)` on renderer init |
| Blank dashboard on load | API returned before skeleton mounted | Ensure `loading` state initialises as `true` |
| Firebase deploy fails | Build folder not found | Run `npm run build` before `firebase deploy` |

---

## 🔗 Related

- [Backend README](../Backend/README.md) — FastAPI server setup and API reference
- [Main README](../README.md) — Full project overview
- [Live App](https://loadiq-smart-ai.web.app/) — Production deployment
