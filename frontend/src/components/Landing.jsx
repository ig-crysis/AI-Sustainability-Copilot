import { useEffect, useMemo, useState } from 'react'
import { Sparkles, BarChart3, PieChart, MapPin, Lightbulb, ShieldCheck, ArrowRight, Leaf, ChevronDown } from 'lucide-react'
import Globe from './Globe'
import LandingSections from './LandingSections'
import './Landing.css'

const FEATURES = [
  { angle: -60,  icon: Sparkles,    title: 'Agentic AI Reasoning',    text: 'An LLM agent decides what to compute for your specific situation, then calls the right tool.' },
  { angle: 0,    icon: BarChart3,   title: 'ML-Trained Footprint Model', text: 'An XGBoost model estimates your carbon footprint in real time from your habits.' },
  { angle: 60,   icon: PieChart,    title: 'Category Breakdown',      text: 'See how transport, food, energy and flights each contribute to your total.' },
  { angle: 120,  icon: MapPin,      title: 'Location-Aware',          text: "Suggestions adapt to your region's grid mix, diet norms and transport options." },
  { angle: 180,  icon: Lightbulb,   title: 'Actionable Guidance',     text: 'Get concrete, personalized suggestions to lower your footprint — not just a number.' },
  { angle: -120, icon: ShieldCheck, title: 'Built Transparently',     text: "Open about how it's trained and its limitations — see the methodology, not a black box." },
]

// Elliptical layout: horizontal room is usually plentiful (limited by hub
// max-width), vertical room is the tight constraint (limited by viewport
// height). Using one radius for both squeezed the left/right cards into
// the globe once the layout got vertically compact — so x and y each get
// their own radius instead.
function nodeOffset(angle, radiusX, radiusY) {
  const rad = (angle * Math.PI) / 180
  return { x: Math.cos(rad) * radiusX, y: Math.sin(rad) * radiusY, dx: Math.cos(rad), dy: Math.sin(rad) }
}

// A jointed, two-knee "spider leg" path from the globe out to each card,
// instead of a plain straight connector.
function legPath(angle, radiusX, radiusY, seed) {
  const { dx, dy, x: ex, y: ey } = nodeOffset(angle, radiusX, radiusY)
  const px = -dy
  const py = dx
  const w1 = 24 * seed
  const w2 = -16 * seed
  const j1x = ex * 0.4 + px * w1
  const j1y = ey * 0.4 + py * w1
  const j2x = ex * 0.72 + px * w2
  const j2y = ey * 0.72 + py * w2
  return {
    d: `M0,0 L${j1x.toFixed(1)},${j1y.toFixed(1)} L${j2x.toFixed(1)},${j2y.toFixed(1)} L${ex.toFixed(1)},${ey.toFixed(1)}`,
    joints: [[j1x, j1y], [j2x, j2y]],
  }
}

const CARD_WIDTH = 200

function useHubLayout() {
  const [layout, setLayout] = useState({ radiusX: 300, radiusY: 190, globeSize: 250 })
  useEffect(() => {
    function compute() {
      const vh = window.innerHeight
      const vw = window.innerWidth
      if (vw <= 900) return
      // Budget the viewport so the CTA stays visible without scrolling:
      // header + tagline (~110px) + cta & its margins (~120px) + section padding (~70px).
      const available = vh - 110 - 120 - 70
      const radiusY = Math.max(135, Math.min(210, available / 2 - 75))
      const globeSize = Math.max(190, Math.min(260, radiusY * 1.3))
      const hubWidth = Math.min(920, vw - 80)
      const radiusX = Math.max(
        globeSize / 2 + CARD_WIDTH / 2 + 24,
        Math.min(340, hubWidth / 2 - CARD_WIDTH / 2 - 16)
      )
      setLayout({ radiusX, radiusY, globeSize })
    }
    compute()
    window.addEventListener('resize', compute)
    return () => window.removeEventListener('resize', compute)
  }, [])
  return layout
}

export default function Landing({ onStart }) {
  const reducedMotion = useMemo(
    () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    []
  )
  const [revealCount, setRevealCount] = useState(reducedMotion ? FEATURES.length : 0)
  const [ctaVisible, setCtaVisible] = useState(reducedMotion)
  const [leaving, setLeaving] = useState(false)
  const { radiusX, radiusY, globeSize } = useHubLayout()
  const hubHalfExtent = Math.max(radiusX, radiusY) + 90

  useEffect(() => {
    if (reducedMotion) return
    const timers = FEATURES.map((_, i) =>
      setTimeout(() => setRevealCount(c => Math.max(c, i + 1)), 900 + i * 420)
    )
    const ctaTimer = setTimeout(() => setCtaVisible(true), 900 + FEATURES.length * 420 + 350)
    return () => { timers.forEach(clearTimeout); clearTimeout(ctaTimer) }
  }, [reducedMotion])

  const handleStart = () => {
    setLeaving(true)
    setTimeout(onStart, 420)
  }

  return (
    <div className={`landing-page ${leaving ? 'leaving' : ''}`}>
      <section className="hero">
        <div className="landing-glow" />

        <header className="landing-header">
          <div className="landing-brand">
            <Leaf size={20} color="#4caf50" />
            <span>AI Sustainability Copilot</span>
          </div>
          <p className="landing-tagline">
            Understand your carbon footprint — powered by a trained ML model and an agentic AI.
          </p>
        </header>

        <div className="hub" style={{ '--radius': `${hubHalfExtent}px` }}>
          <div className="globe-slot">
            <Globe size={globeSize} />
          </div>

          <svg
            className="connectors"
            viewBox={`${-hubHalfExtent} ${-hubHalfExtent} ${hubHalfExtent * 2} ${hubHalfExtent * 2}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {FEATURES.map((f, i) => {
              const revealed = i < revealCount
              const seed = i % 2 === 0 ? 1 : -1
              const { d, joints } = legPath(f.angle, radiusX, radiusY, seed)
              return (
                <g key={f.title} className={`leg ${revealed ? 'drawn' : ''}`}>
                  <path d={d} className="connector" />
                  {joints.map(([jx, jy], k) => (
                    <circle key={k} cx={jx} cy={jy} r="2.6" className="joint" />
                  ))}
                </g>
              )
            })}
          </svg>

          {FEATURES.map((f, i) => {
            const { x, y } = nodeOffset(f.angle, radiusX, radiusY)
            const Icon = f.icon
            const revealed = i < revealCount
            return (
              <div
                key={f.title}
                className={`feature-node ${revealed ? 'revealed' : ''}`}
                style={{ '--nx': `${x}px`, '--ny': `${y}px` }}
              >
                <div className="feature-icon"><Icon size={17} /></div>
                <div className="feature-copy">
                  <h3>{f.title}</h3>
                  <p>{f.text}</p>
                </div>
              </div>
            )
          })}
        </div>

        <button
          className={`cta ${ctaVisible ? 'visible' : ''}`}
          onClick={handleStart}
          disabled={!ctaVisible}
        >
          Let's Get Started <ArrowRight size={20} />
        </button>

        <div className="scroll-hint">
          <span>Why this exists</span>
          <ChevronDown size={16} />
        </div>
      </section>

      <LandingSections onStart={handleStart} />
    </div>
  )
}
