<a id="top"></a>

<div align="center">

# PicoTuri - EditJudge

### _Real-time algorithm quality assessment and benchmarking dashboard_

<img src="https://img.shields.io/badge/PicoTuri-Algorithm_Dashboard-7C3AED?style=for-the-badge" alt="PicoTuri badge" />
<img src="https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=000000" alt="React badge" />
<img src="https://img.shields.io/badge/Flask-2.3.3-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask badge" />
<img src="https://img.shields.io/badge/Vite-7.1-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite badge" />
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT badge" />

[Live Demo](https://pico-turi-edit-judge.vercel.app/) • [Repository](https://github.com/mangeshraut712/PicoTuri-EditJudge)

**[About](#about) • [Highlights](#highlights) • [Tech Stack](#tech-stack) • [Quick Start](#quick-start) • [Project Structure](#project-structure) • [Scripts](#scripts) • [License](#license) • [Contact](#contact)**

</div>

---

## 📖 Table of Contents

- [About](#about)
- [Highlights](#highlights)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [License](#license)
- [Contact](#contact)

---

<a id="about"></a>

## About

PicoTuri-EditJudge is a React + Flask dashboard for comparing algorithm quality, watching live metrics, and visualizing benchmark results. It is built around chart-driven pages, lightweight API endpoints, and a small set of deterministic test responses that make the experience easy to demo and easy to extend.

<a id="highlights"></a>

## Highlights

- Seven algorithm views covering quality scoring, diffusion, DPO, Core ML, baseline, and feature extraction workflows.
- Interactive charts and dashboards built with Recharts and animated UI sections.
- A lightweight Flask API with health, stats, and test endpoints for local and Vercel deployment.
- Jest-based frontend testing with snapshot and coverage commands.

<a id="tech-stack"></a>

## Tech Stack

**Frontend**

- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Recharts

**Backend**

- Flask 2.3
- Flask-CORS
- NumPy
- Pillow
- Vercel WSGI

**Tooling**

- Jest
- Babel
- Pyright
- Vercel
- GitHub-hosted deployment

<a id="quick-start"></a>

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.8+
- npm and `pip`

### Install

```bash
git clone https://github.com/mangeshraut712/PicoTuri-EditJudge.git
cd PicoTuri-EditJudge
npm install
pip install -r api/requirements.txt
```

### Run

```bash
# Frontend
npm run dev

# API
python api/index.py
```

The frontend runs on Vite, while the Flask API listens on port `5001`.

<a id="project-structure"></a>

## Project Structure

```text
PicoTuri-EditJudge/
├── src/                # React pages, components, and utilities
├── api/                # Flask app and Vercel handler
├── public/             # Static assets
├── package.json        # Frontend scripts and dependencies
├── requirements.txt    # Python dependency list
└── vercel.json         # Deployment configuration
```

<a id="scripts"></a>

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite dev server |
| `npm run build` | Build the frontend for production |
| `npm run preview` | Preview the production build locally |
| `npm run start` | Start the Vite server alias |
| `npm run test` | Run Jest in watch mode |
| `npm run test:coverage` | Generate test coverage output |
| `npm run test:update` | Refresh Jest snapshots |
| `python api/index.py` | Start the Flask API on port 5001 |

<a id="license"></a>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

<a id="contact"></a>

## Contact

- Live demo: [pico-turi-edit-judge.vercel.app](https://pico-turi-edit-judge.vercel.app/)
- Repository issues: [mangeshraut712/PicoTuri-EditJudge/issues](https://github.com/mangeshraut712/PicoTuri-EditJudge/issues)

[↑ Back to Top](#top)
