import React, { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Navigation from './components/layout/Navigation'
import AlgorithmsPage from './pages/AlgorithmsPage'
import PerformanceDashboard from './pages/PerformanceDashboard'
import ResearchPage from './pages/ResearchPage'

function App() {
  const [navigationOpen, setNavigationOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation Sidebar */}
      <Navigation
        isOpen={navigationOpen}
        onToggle={setNavigationOpen}
      />

      {/* Main Content */}
      <Routes>
        <Route path="/" element={<AlgorithmsPage />} />
        <Route path="/performance" element={<PerformanceDashboard />} />
        <Route path="/research" element={<ResearchPage />} />
      </Routes>
    </div>
  )
}

export default App
