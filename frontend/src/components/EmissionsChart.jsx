import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const COLORS = ['#4caf50', '#81c784', '#ff7043', '#ffb74d']

export default function EmissionsChart({ data }) {
  if (!data) return (
    <div style={{ color: '#4a7a4a', fontSize: 13, textAlign: 'center', padding: '24px 0' }}>
      Ask about your footprint to see the breakdown chart
    </div>
  )

  const chartData = [
    { name: 'Transport', value: data.transport },
    { name: 'Energy',    value: data.energy },
    { name: 'Food',      value: data.food },
    { name: 'Flights',   value: data.flights },
  ].filter(d => d.value > 0)

  const total = data.total
  ? data.total.toFixed(1)
  : chartData.reduce((s, d) => s + d.value, 0).toFixed(1)

  return (
    <div>
      <div style={{ fontSize: 12, color: '#81c784', marginBottom: 8 }}>
        Monthly CO₂: <strong style={{ color: '#e8f5e9' }}>{total} kg</strong>
      <span style={{ fontSize: 11, color: '#4a7a4a', marginLeft: 6 }}>(model output)</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie data={chartData} cx="50%" cy="50%" innerRadius={50}
            outerRadius={80} paddingAngle={3} dataKey="value">
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v) => [`${v} kg CO₂`, '']}
            contentStyle={{ background: '#1a2e1a', border: '1px solid #2e7d32', borderRadius: 8, fontSize: 12 }}
          />
          <Legend iconSize={10} wrapperStyle={{ fontSize: 12, color: '#81c784' }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}