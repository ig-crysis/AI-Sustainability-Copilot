const CONFIG = {
  'THRESHOLD: LOW': {
    color: '#4caf50', bg: 'rgba(76,175,80,0.12)',
    border: 'rgba(76,175,80,0.3)',
    icon: '🌿', label: 'Low Footprint',
    message: "You're below the global average. Keep it up!"
  },
  'THRESHOLD: MODERATE': {
    color: '#ffb74d', bg: 'rgba(255,183,77,0.12)',
    border: 'rgba(255,183,77,0.3)',
    icon: '⚠️', label: 'Moderate Footprint',
    message: "Slightly above ideal. A few small changes can help."
  },
  'THRESHOLD: HIGH': {
    color: '#ff7043', bg: 'rgba(255,112,67,0.12)',
    border: 'rgba(255,112,67,0.3)',
    icon: '🔴', label: 'High Footprint',
    message: "Above global average. Action recommended."
  },
  'THRESHOLD: CRITICAL': {
    color: '#f44336', bg: 'rgba(244,67,54,0.12)',
    border: 'rgba(244,67,54,0.3)',
    icon: '🚨', label: 'Critical Footprint',
    message: "Well above global average. Urgent action needed."
  },
}

export default function ThresholdBadge({ threshold }) {
  const cfg = CONFIG[threshold]
  if (!cfg) return null

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      background: cfg.bg, border: `1px solid ${cfg.border}`,
      borderRadius: 12, padding: '10px 16px', margin: '0 24px 12px',
    }}>
      <span style={{ fontSize: 18 }}>{cfg.icon}</span>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: cfg.color }}>
          {cfg.label}
        </div>
        <div style={{ fontSize: 12, color: '#a5d6a7', marginTop: 2 }}>
          {cfg.message}
        </div>
      </div>
    </div>
  )
}