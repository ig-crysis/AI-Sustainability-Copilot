import { useEffect, useRef, useState } from 'react'
import {
  Target, Cpu, TrendingDown, MessageSquare, BarChart3,
  CheckCircle2, ArrowRight, Gauge, GitBranch, Layers,
} from 'lucide-react'
import './LandingSections.css'

function useReveal() {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) { setInView(true); obs.disconnect() }
      },
      { threshold: 0.18 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return [ref, inView]
}

const WHY = [
  {
    icon: Target,
    title: 'Generic calculators fall short',
    text: 'Most footprint tools use flat, one-size-fits-all averages that ignore your actual habits, diet and location.',
  },
  {
    icon: Cpu,
    title: 'Reasoning, not a rigid form',
    text: 'An LLM agent interprets what you actually say, extracts the right inputs, and decides what to compute — no dropdowns required.',
  },
  {
    icon: TrendingDown,
    title: 'Grounded in a trained model',
    text: 'Estimates come from an XGBoost model, not a hardcoded lookup table — numbers reflect real patterns learned from data.',
  },
]

const STEPS = [
  { icon: MessageSquare, title: 'Tell it about your day', text: 'Transport, food, energy, flights — in your own words.' },
  { icon: Cpu,            title: 'The agent extracts & reasons', text: 'It structures your message and decides what to calculate.' },
  { icon: BarChart3,      title: 'The model estimates', text: 'A trained XGBoost model produces a category-by-category breakdown.' },
  { icon: CheckCircle2,   title: 'You get real guidance', text: 'Concrete, personalized suggestions — not just a single number.' },
]

const STATS = [
  { icon: Gauge,     value: 'R² 0.83', label: 'Model fit on held-out data' },
  { icon: Layers,    value: 'XGBoost', label: 'Gradient-boosted footprint model' },
  { icon: Cpu,       value: 'Llama 3.1 8B', label: 'Reasoning agent via Groq' },
  { icon: GitBranch, value: '2-call pipeline', label: 'Extract → estimate, not a black box' },
]

export default function LandingSections({ onStart }) {
  const [whyRef, whyIn] = useReveal()
  const [stepsRef, stepsIn] = useReveal()
  const [statsRef, statsIn] = useReveal()
  const [ctaRef, ctaIn] = useReveal()

  return (
    <div className="landing-sections">
      <section ref={whyRef} className={`section why-section ${whyIn ? 'in-view' : ''}`}>
        <div className="section-heading">
          <span className="eyebrow">Why this exists</span>
          <h2>Carbon numbers you can actually trust</h2>
          <p>Most "carbon calculators" are static spreadsheets in disguise. This one reasons about what you tell it, then grounds every number in a model trained for the job.</p>
        </div>
        <div className="why-grid">
          {WHY.map((w, i) => {
            const Icon = w.icon
            return (
              <div key={w.title} className="why-card" style={{ '--i': i }}>
                <div className="why-icon"><Icon size={20} /></div>
                <h3>{w.title}</h3>
                <p>{w.text}</p>
              </div>
            )
          })}
        </div>
      </section>

      <section ref={stepsRef} className={`section steps-section ${stepsIn ? 'in-view' : ''}`}>
        <div className="section-heading">
          <span className="eyebrow">How it works</span>
          <h2>From a sentence to a breakdown</h2>
        </div>
        <div className="steps-row">
          {STEPS.map((s, i) => {
            const Icon = s.icon
            return (
              <div key={s.title} className="step-card" style={{ '--i': i }}>
                <div className="step-number">{i + 1}</div>
                <div className="step-icon"><Icon size={20} /></div>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
                {i < STEPS.length - 1 && <ArrowRight className="step-arrow" size={18} />}
              </div>
            )
          })}
        </div>
      </section>

      <section ref={statsRef} className={`section stats-section ${statsIn ? 'in-view' : ''}`}>
        <div className="section-heading">
          <span className="eyebrow">Under the hood</span>
          <h2>Built transparently, on purpose</h2>
          <p>The methodology, training data and limitations are documented — not hidden behind marketing numbers.</p>
        </div>
        <div className="stats-grid">
          {STATS.map((s, i) => {
            const Icon = s.icon
            return (
              <div key={s.label} className="stat-tile" style={{ '--i': i }}>
                <Icon size={18} className="stat-icon" />
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            )
          })}
        </div>
      </section>

      <section ref={ctaRef} className={`section final-cta ${ctaIn ? 'in-view' : ''}`}>
        <h2>See your own footprint</h2>
        <p>It takes one sentence about your day to get a real, personalized breakdown.</p>
        <button className="cta cta-final" onClick={onStart}>
          Let's Get Started <ArrowRight size={20} />
        </button>
      </section>
    </div>
  )
}
