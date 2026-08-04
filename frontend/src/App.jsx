import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import EmissionsChart from './components/EmissionsChart'
import ToolTrace from './components/ToolTrace'
import Sidebar from './components/Sidebar'
import ThresholdBadge from './components/ThresholdBadge'
import { Leaf, Send, Loader } from 'lucide-react'
import { sendMessage, getUserLocation } from './api'
import './App.css'

// ── Constants (outside component is correct) ─────────────────────────────────
const SESSION_ID = `session_${Date.now()}`

const SUGGESTIONS = [
  "I drive a petrol car 30km daily, eat chicken, use 15kWh/day in India. What's my footprint?",
  "Compare flying vs taking a train for 500km travel",
  "I ride a bike everyday for 2km and use mobile phone for 7hrs a day,",
  "What's the carbon impact of eating non-veg everyday in Europe",
]

// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  // ALL hooks must be here, inside the function
  const [messages,    setMessages]  = useState([])
  const [input,       setInput]     = useState('')
  const [loading,     setLoading]   = useState(false)
  const [toolSteps,   setToolSteps] = useState([])
  const [chartData,   setChartData] = useState(null)
  const [sidebarOpen, setSidebar]   = useState(true)
  const [threshold,   setThreshold] = useState(null)   // ← moved here
  const bottomRef                   = useRef(null)
  const [userLocation, setUserLocation] = useState(null)
  const [locationStatus, setLocationStatus] = useState('pending') // pending | granted | denied


  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    setMessages([{
      role: 'assistant',
      content: "Hi! I'm your AI Sustainability Copilot 🌱\n\nI use a trained XGBoost model + agentic AI to analyze your carbon footprint in real time. Tell me about your daily habits — transport, food, energy usage — and I'll give you a detailed breakdown with actionable suggestions.\n\nTry one of the examples below to get started!",
      tools: [],
    }])
  }, [])

  useEffect(() => {
  getUserLocation().then(loc => {
    if (loc) {
        setUserLocation(loc)
        setLocationStatus('granted')
        console.log('[Location]', loc)
      } else {
        setLocationStatus('denied')
      }
    })
  }, [])

  const handleSend = async (text) => {
    const msg = text || input.trim()
  if (!msg || loading) return

  setInput('')
  setChartData(null)
  setThreshold(null)
  setMessages(prev => [...prev, { role: 'user', content: msg }])
  setLoading(true)
  setToolSteps([])

  // Inject location context into message if available
  let augmentedMsg = msg
  if (userLocation) {
    augmentedMsg = `${msg}\n\n[USER LOCATION: ${userLocation.city}, ${userLocation.state}, ${userLocation.country} (${userLocation.country_code})]`
  }

  try {
    const { data } = await sendMessage(augmentedMsg, SESSION_ID, userLocation)

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        tools: data.tools_called,
      }])

      setToolSteps(data.tools_called)
      setThreshold(data.threshold || null)

      // Use the backend's own per-category breakdown (same numbers the
      // synthesis LLM's response text is grounded in) rather than
      // re-deriving them client-side from a separate, easily-drifting set
      // of hardcoded emission factors and tool-arg field names.
      if (data.breakdown) {
        const b = data.breakdown
        setChartData({
          transport: b.transport_kg || 0,
          food:      b.food_kg || 0,
          energy:    b.energy_kg || 0,
          flights:   b.flights_kg || 0,
          total:     data.actual_co2,
        })
      }

    } catch (err) {
      console.error('[chat] request failed:', err)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ Something went wrong: ${err.response?.data?.detail || err.message}`,
        tools: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="app-layout">
      <Sidebar open={sidebarOpen} onToggle={() => setSidebar(!sidebarOpen)}
        toolSteps={toolSteps} chartData={chartData} />

      <div className="main-area">
        <header className="header glass">
          <div className="header-left">
            <Leaf size={22} color="#4caf50" />
            <span className="header-title">AI Sustainability Copilot</span>
            <span className="header-badge">XGBoost + Groq LLM</span>
          </div>
          <div className="header-right">
  <span className="status-dot" />
  <span style={{ fontSize: 13, color: '#81c784' }}>Live</span>
  {userLocation && (
    <span style={{
      fontSize: 11, color: '#4a7a4a',
      background: 'rgba(76,175,80,0.08)',
      border: '1px solid rgba(76,175,80,0.2)',
      padding: '2px 8px', borderRadius: 20, marginLeft: 8,
    }}>
      📍 {userLocation.city || userLocation.state}, {userLocation.country}
    </span>
  )}
  {locationStatus === 'denied' && (
    <span style={{ fontSize: 11, color: '#4a7a4a', marginLeft: 8 }}>
      📍 Location off
    </span>
  )}
</div>
        </header>

        <div className="messages-area">
          {messages.map((msg, i) => (
            <ChatMessage key={i} role={msg.role} content={msg.content} />
          ))}
          {loading && (
            <div className="thinking-row">
              <Loader size={16} className="spin" color="#4caf50" />
              <span>Analyzing with AI agent...</span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {messages.length <= 1 && (
          <div className="suggestions">
            {SUGGESTIONS.map((s, i) => (
              <button key={i} className="suggestion-chip" onClick={() => handleSend(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        <ThresholdBadge threshold={threshold} />

        <div className="input-area glass">
          <textarea
            className="input-box"
            placeholder="Ask about your carbon footprint..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
          >
            {loading ? <Loader size={18} className="spin" /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </div>
  )
}