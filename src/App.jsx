import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import AlgorithmsPage from './pages/AlgorithmsPage'
import ResearchPage from './pages/ResearchPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AlgorithmsPage />} />
        <Route path="/research" element={<ResearchPage />} />
      </Routes>
    </Router>
  )
}

export default App
