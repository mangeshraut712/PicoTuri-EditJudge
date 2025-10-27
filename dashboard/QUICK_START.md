# 🚀 Quick Start - Advanced Dashboard 2025

## ⚡ Fastest Way to Run

```bash
# 1. Navigate to dashboard
cd /Users/mangeshraut/Downloads/PicoTuri-EditJudge/dashboard

# 2. Install dependencies (first time only)
npm install

# 3. Start dashboard
npm run dev
```

Then open: **http://localhost:3000**

## 📋 Prerequisites

Make sure Python backend is running:

```bash
# In another terminal, from project root
python src/gui/web_dashboard.py
```

Backend runs on: **http://localhost:5001**

## ✅ What You Get

- 🎨 Modern React dashboard
- 📊 Interactive charts with Recharts
- 🔄 Real-time algorithm testing
- 📈 Accuracy visualizations
- ⚡ Lightning-fast Vite dev server
- 🎭 Smooth animations with Framer Motion

## 🎯 Features

### All 7 Algorithms Integrated
1. **Quality Scorer** - Component breakdown charts
2. **Diffusion Model** - Parameter visualization
3. **DPO Training** - Loss curves
4. **Multi-Turn Editor** - Success rate graphs
5. **Core ML Optimizer** - Performance metrics
6. **Baseline Model** - Pipeline stats
7. **Feature Extraction** - TF-IDF analysis

### Charts & Visualizations
- Circular progress indicators
- Bar charts for components
- Line graphs for trends
- Radar charts for multi-metrics
- Pie charts for distributions
- Real-time updates

## 🔧 Troubleshooting

### Port 3000 in use?
```bash
# Use different port
vite --port 3001
```

### Backend not responding?
```bash
# Check if backend is running
curl http://localhost:5001/api/stats
```

### Dependencies fail?
```bash
# Clear and reinstall
rm -rf node_modules
npm install
```

## 📦 What's Installed

- React 18.2.0
- Vite 5.0.8
- Tailwind CSS 3.3.6
- Recharts 2.10.3
- Framer Motion 10.16.16
- Lucide React 0.294.0
- Axios 1.6.2

## 🎨 Customization

All configuration files are ready:
- `package.json` - Dependencies
- `vite.config.js` - Dev server & proxy
- `tailwind.config.js` - Styling
- `index.html` - HTML template

## 📊 Dashboard Structure

```
Dashboard (http://localhost:3000)
├── Header - Project title & stats
├── Stats Cards - 7/7 algorithms, 100% quality
├── Algorithm Grid - 7 interactive cards
│   ├── Quality Scorer
│   ├── Diffusion Model
│   ├── DPO Training
│   ├── Multi-Turn Editor
│   ├── Core ML Optimizer
│   ├── Baseline Model
│   └── Feature Extraction
└── Charts Section - Real-time visualizations
```

## 🚀 Next Steps

1. **Install**: `npm install`
2. **Run**: `npm run dev`
3. **Test**: Click any algorithm card
4. **View**: See charts and metrics
5. **Customize**: Edit components as needed

---

**Ready in 3 commands!** 🎉
