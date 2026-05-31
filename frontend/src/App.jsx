import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import PredictionEngine from './pages/PredictionEngine'
import Architecture from './pages/Architecture'
import NotebookView from './pages/NotebookView'
import ReportGenerator from './pages/ReportGenerator'

const API = import.meta.env.VITE_API_URL || ''

function App() {
  return (
    <>
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>Manulife Task</h1>
        </div>

        <div className="sidebar-nav">
          <div className="nav-section-label">Part 1 - Modeling</div>
          <NavLink to="/predict" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">&#9881;</span>
            <span>Prediction Engine</span>
          </NavLink>
          <NavLink to="/notebook" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">&#128209;</span>
            <span>Notebook Walkthrough</span>
          </NavLink>

          <div className="nav-section-label">Part 2 - GenAI</div>
          <NavLink to="/architecture" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">&#9878;</span>
            <span>Architecture</span>
          </NavLink>
          <NavLink to="/report" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">&#128196;</span>
            <span>Report Generator</span>
          </NavLink>
        </div>

        <div className="sidebar-footer">
          LTC Pure Premium Model v2<br />
          Frequency x Severity + Tweedie
        </div>
      </nav>

      {/* Main */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/predict" replace />} />
          <Route path="/predict" element={<PredictionEngine api={API} />} />
          <Route path="/notebook" element={<NotebookView />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/report" element={<ReportGenerator api={API} />} />
        </Routes>
      </main>
    </>
  )
}

export default App
