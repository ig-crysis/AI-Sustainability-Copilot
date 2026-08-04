import { PanelLeftClose, PanelLeft, Cpu, BarChart2 } from 'lucide-react'
import EmissionsChart from './EmissionsChart'
import ToolTrace from './ToolTrace'
import './Sidebar.css'

export default function Sidebar({ open, onToggle, toolSteps, chartData }) {
  return (
    <aside className={`sidebar glass ${open ? 'open' : 'closed'}`}>
      <button className="toggle-btn" onClick={onToggle}>
        {open ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
      </button>

      {open && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">
              <BarChart2 size={14} color="#4caf50" />
              Emissions Breakdown
            </div>
            <EmissionsChart data={chartData} />
          </div>

          <div className="sidebar-divider" />

          <div className="sidebar-section">
            <div className="sidebar-label">
              <Cpu size={14} color="#4caf50" />
              Agent Tool Trace
            </div>
            <ToolTrace steps={toolSteps} />
          </div>

          <div className="sidebar-divider" />

          <div className="sidebar-section">
            <div className="sidebar-label" style={{ marginBottom: 8 }}>Model Info</div>
            <div className="model-info">
              <div className="info-row">
                <span>Model</span><span>XGBoost</span>
              </div>
              <div className="info-row">
                <span>R² Score</span><span style={{ color: '#4caf50' }}>0.83</span>
              </div>
              <div className="info-row">
                <span>LLM</span><span>Llama 3.1 8B (Groq)</span>
              </div>
              <div className="info-row">
                <span>Agent</span><span>2-call pipeline</span>
              </div>
            </div>
          </div>
        </>
      )}
    </aside>
  )
}