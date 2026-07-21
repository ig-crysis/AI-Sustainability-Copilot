import './ToolTrace.css'

const TOOL_ICONS = {
  predict_footprint:          '🤖',
  get_live_carbon_intensity:  '⚡',
  compare_transport_scenarios:'🚗',
  get_regional_baseline:      '🌍',
}

export default function ToolTrace({ steps }) {
  if (!steps?.length) return (
    <div style={{ color: '#4a7a4a', fontSize: 13, textAlign: 'center', padding: '16px 0' }}>
      Tool calls will appear here during analysis
    </div>
  )

  return (
    <div className="tool-trace">
      {steps.map((step, i) => (
        <div key={i} className="tool-step">
          <div className="tool-header">
            <span>{TOOL_ICONS[step.tool] || '🔧'}</span>
            <span className="tool-name">{step.tool}</span>
          </div>
          <div className="tool-input">
            {Object.entries(step.input).map(([k, v]) => (
              <div key={k} className="tool-param">
                <span className="param-key">{k}</span>
                <span className="param-val">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}