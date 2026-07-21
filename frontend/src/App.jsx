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

const TRANSPORT_EF = {
  car_petrol: 0.192, car_diesel: 0.171, car_ev: 0.053,
  bus: 0.089, train: 0.041, flight_short: 0.255,
  flight_long: 0.195, motorcycle: 0.114, bicycle: 0.0, walking: 0.0,
}
const FOOD_EF = {
  beef: 27.0, lamb: 39.2, pork: 12.1, chicken: 6.9,
  fish: 6.1, eggs: 4.8, dairy: 3.2, rice: 4.0,
  vegetables: 2.0, fruits: 1.1, legumes: 0.9, nuts: 2.5,
}
const ENERGY_EF = {
  coal: 0.820, natural_gas: 0.490, oil: 0.650,
  solar: 0.041, wind: 0.011, hydro: 0.024, nuclear: 0.012,
  grid_india: 0.708, grid_us: 0.386, grid_eu: 0.276,
}

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

      const predictStep = data.tools_called.find(t => t.tool === 'predict_footprint')
if (predictStep) {
  const inp = predictStep.input
  const transport = +((inp.km_per_day || 0) * 30 *
    (TRANSPORT_EF[inp.transport_type] ?? 0.1)).toFixed(1)
  const food = (inp.kg_food_per_day || 0) > 0.01
    ? +((inp.kg_food_per_day || 0) * 30 * (FOOD_EF[inp.food_type] ?? 3.0)).toFixed(1)
    : 0
  const energy = +((inp.kwh_per_day || 0) * 30 *
    (ENERGY_EF[inp.energy_source] ?? 0.5)).toFixed(1)
  const flights = +((inp.flight_km_total || 0) / 12 * 0.195).toFixed(1)
  // Use actual model CO2 as total if available, else sum of components
  const total = data.actual_co2 || (transport + food + energy + flights)
  setChartData({ transport, food, energy, flights, total })
}

    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Something went wrong. Make sure the backend is running on port 8000.',
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
            <span className="header-badge">XGBoost + LangGraph</span>
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