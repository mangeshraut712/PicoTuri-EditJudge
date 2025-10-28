<div align="center">
  <h1>🎯 PicoTuri-EditJudge</h1>
  <p><strong>Interactive dashboard for exploring mock AI algorithm telemetry</strong></p>
  <p>
    <a href="#getting-started">Getting Started</a> ·
    <a href="#api-overview">API Overview</a> ·
    <a href="#frontend-experiences">Frontend Experiences</a>
  </p>
</div>

PicoTuri-EditJudge pairs a lightweight Flask API with a Vite + React dashboard. The project provides mock responses for seven AI-related algorithms so designers and engineers can experiment with data visualisation and UX flows without needing live infrastructure.

> **Note**: The repository intentionally ships with mock data. None of the endpoints call external services. This keeps the project easy to run locally and ready for deployment to Vercel or any static hosting provider.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [API Overview](#api-overview)
5. [Frontend Experiences](#frontend-experiences)
6. [Development Commands](#development-commands)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

## Features

- 🔁 **Create-React App Replacement** – Modern Vite toolchain with React 18, Tailwind, and lucide icons.
- 📊 **Rich Algorithm Visualisations** – Radar, bar, pie, and line charts powered by Recharts.
- 🧪 **Seven Mock Algorithms** – Quality scorer, diffusion model, DPO training, multi-turn editor, Core ML optimizer, baseline model, and feature extraction.
- 🚦 **Status Dashboards** – Algorithms page, performance dashboard, and research hub to demonstrate design patterns.
- 🧰 **Deployable API** – Flask app exposes JSON endpoints that the dashboard consumes. Safe to host on Vercel via `vercel_wsgi`.

## Project Structure

```
PicoTuri-EditJudge/
├── api/                    # Flask API (index.py, requirements, handler)
├── public/                 # Static assets served by Vite
├── src/                    # React source code
│   ├── components/
│   │   ├── algorithms/     # Reusable charts & cards
│   │   └── layout/         # Navigation shell
│   ├── pages/              # Algorithms, Performance Dashboard, Research Hub
│   ├── main.jsx            # React entry point
│   └── index.css           # Tailwind layers + custom styles
├── src_main/               # Archived ML notebooks & scripts (optional)
├── package.json
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Getting Started

### Prerequisites

- Node.js 18+ (Vite requirement)
- Python 3.10+ (tested with 3.12)
- `pip` and `virtualenv` (recommended)

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

### 3. Run the Flask API

```bash
cd api
source .venv/bin/activate
python index.py
```

The API listens on `http://localhost:5001` by default. If the port is already used, edit `app.run(... port=5002)` in `api/index.py`.

### 4. Run the React Dashboard

```bash
npm run dev
```

Visit the URL printed in the terminal (typically `http://localhost:5173`).

### 5. Production Build

```bash
npm run build
```

The output appears in `dist/`. Deploy to any static host or connect to Vercel for zero-config hosting.

## API Overview

All responses are deterministic mock payloads designed to feed the UI. Endpoints live under `/api`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health probe returning service name and timestamp. |
| `/api/stats` | GET | High level usage metrics consumed by the dashboard. |
| `/api/performance/status` | GET | System performance data (uptime, CPU, memory, requests). |
| `/api/test/quality-scorer` | POST | Component scores, weights, overall grade. |
| `/api/test/diffusion-model` | POST | Parameters, architecture, inference timings. |
| `/api/test/dpo-training` | POST | Loss, preference accuracy, step counts. |
| `/api/test/multi-turn` | POST | Session success metrics for conversational editing. |
| `/api/test/coreml` | POST | Core ML conversion stats, size reduction, target platform. |
| `/api/test/baseline` | POST | Classifier metadata and evaluation metrics. |
| `/api/test/features` | POST | Text/vision similarity indicators for feature extraction. |

### Deploying on Vercel

The API automatically exposes a `handler` via `vercel_wsgi`. No extra configuration is required—deploy the `api` directory as a Python serverless function and the `dist` directory as static assets.

## Frontend Experiences

- **Algorithms Page**: Run each algorithm, view responsive cards, and inspect detailed modal charts powered by `ChartVisualization.jsx`.
- **Performance Dashboard**: Real-time style charts highlighting CPU usage, request throughput, algorithm distribution, and system uptime.
- **Research Hub**: Curated set of research articles and experiments with filters and badges to demonstrate complex layout patterns.

All pages share the navigation shell in `src/components/layout/Navigation.jsx`. Tailwind plus a custom glassmorphism theme drives the visual language.

## Development Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start the Vite development server with hot module replacement. |
| `npm run build` | Generate a production build in `dist/`. |
| `npm run preview` | Preview the production build locally. |
| `python -m py_compile api/index.py` | Quick syntax check for the Flask API. |

## Troubleshooting

- **Port already in use (5001)**: Stop other services or change the port in `api/index.py`.
- **Missing dependencies**: Ensure you activated the virtual environment before installing backend packages.
- **CORS issues**: `api/index.py` enables development CORS by default. Confirm the console log says `CORS enabled for development`
- **Frontend uses stale data**: Clear local storage or refresh after restarting the Flask server—the mock endpoints are stateless.

## License

This project is released under the [Apache 2.0 License](LICENSE).
