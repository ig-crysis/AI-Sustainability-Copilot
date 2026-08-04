import { useEffect, useRef } from 'react'

// Rough continent silhouettes as [lat, lon] polygons — hand-approximated for
// a small decorative globe, not survey-accurate GIS data. Good enough to
// read unmistakably as "Earth" once projected onto a rotating sphere.
const AFRICA = [[37,-5],[32,-9],[14,-17],[4,-8],[-3,9],[-18,12],[-34,19],[-29,31],[-11,40],[3,41],[12,43],[22,38],[31,32],[37,-5]]
const EUROPE = [[36,-9],[43,-9],[51,-5],[58,5],[60,25],[55,40],[45,40],[38,27],[36,20],[36,-9]]
const ASIA = [[45,25],[60,40],[70,60],[75,90],[70,140],[60,160],[45,140],[30,120],[10,100],[5,80],[12,60],[25,50],[35,35],[45,25]]
const NORTH_AMERICA = [[70,-160],[70,-90],[60,-65],[45,-60],[30,-80],[18,-95],[15,-105],[30,-117],[48,-125],[60,-140],[70,-160]]
const SOUTH_AMERICA = [[12,-72],[10,-62],[-5,-35],[-23,-43],[-34,-58],[-55,-68],[-38,-73],[-18,-70],[-2,-79],[12,-72]]
const AUSTRALIA = [[-11,130],[-12,142],[-20,149],[-28,153],[-38,147],[-35,138],[-32,115],[-20,113],[-11,130]]
const GREENLAND = [[83,-35],[76,-20],[60,-45],[70,-55],[83,-35]]
const CONTINENTS = [AFRICA, EUROPE, ASIA, NORTH_AMERICA, SOUTH_AMERICA, AUSTRALIA, GREENLAND]

function pointInPolygon(lat, lon, poly) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [latI, lonI] = poly[i]
    const [latJ, lonJ] = poly[j]
    const crosses = (lonI > lon) !== (lonJ > lon) &&
      lat < ((latJ - latI) * (lon - lonI)) / (lonJ - lonI) + latI
    if (crosses) inside = !inside
  }
  return inside
}

function buildLandPoints() {
  const pts = []
  for (let lat = -64; lat <= 82; lat += 2.6) {
    for (let lon = -180; lon < 180; lon += 2.6) {
      for (const poly of CONTINENTS) {
        if (pointInPolygon(lat, lon, poly)) { pts.push([lat, lon]); break }
      }
    }
  }
  return pts
}
const LAND_POINTS = buildLandPoints()

const GRATICULE_LATS = [-60, -30, 0, 30, 60]
const GRATICULE_LONS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
const TILT = -16 * (Math.PI / 180)
const SIN_T = Math.sin(TILT)
const COS_T = Math.cos(TILT)

// Projects a lat/lon (deg) to screen space given the current rotation and
// sphere geometry. Returns null if the point is on the far side of the sphere.
function project(lat, lon, rot, R, cx, cy) {
  const latR = (lat * Math.PI) / 180
  const lonR = (lon * Math.PI) / 180 + rot
  const x = Math.cos(latR) * Math.sin(lonR)
  const y0 = Math.sin(latR)
  const z0 = Math.cos(latR) * Math.cos(lonR)
  const y = y0 * COS_T - z0 * SIN_T
  const z = y0 * SIN_T + z0 * COS_T
  if (z < -0.02) return null
  return { px: cx + x * R, py: cy - y * R, depth: (z + 1) / 2 }
}

export default function Globe({ size = 260 }) {
  const canvasRef = useRef(null)
  const angleRef = useRef(0)
  const sizeRef = useRef(size)

  // Keep the backing canvas in sync with layout changes, independent of
  // the animation loop below — resize churn must never restart rotation.
  useEffect(() => {
    sizeRef.current = size
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0)
  }, [size])

  // Animation loop: set up exactly once and left running for the life of
  // the component, reading sizeRef each frame so it never needs to restart.
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    function strokeGraticuleLine(points) {
      ctx.beginPath()
      let started = false
      for (const p of points) {
        if (!p) { started = false; continue }
        if (!started) { ctx.moveTo(p.px, p.py); started = true }
        else ctx.lineTo(p.px, p.py)
      }
      ctx.stroke()
    }

    let raf
    function draw() {
      const s = sizeRef.current
      const R = s / 2 - 6
      const cx = s / 2
      const cy = s / 2
      const rot = angleRef.current

      ctx.clearRect(0, 0, s, s)

      // ocean sphere
      const grad = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.42, R * 0.08, cx, cy, R)
      grad.addColorStop(0, 'rgba(165, 232, 168, 0.38)')
      grad.addColorStop(0.55, 'rgba(76, 175, 80, 0.22)')
      grad.addColorStop(1, 'rgba(4, 18, 9, 0.32)')
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.fillStyle = grad
      ctx.fill()

      // graticule (lat/lon grid) — sells the 3D curvature as it rotates
      ctx.strokeStyle = 'rgba(200, 245, 203, 0.16)'
      ctx.lineWidth = 1
      for (const lat of GRATICULE_LATS) {
        const pts = []
        for (let lon = -180; lon <= 180; lon += 4) pts.push(project(lat, lon, rot, R, cx, cy))
        strokeGraticuleLine(pts)
      }
      for (const lon of GRATICULE_LONS) {
        const pts = []
        for (let lat = -90; lat <= 90; lat += 4) pts.push(project(lat, lon, rot, R, cx, cy))
        strokeGraticuleLine(pts)
      }

      // landmasses — dark green dots, brighter/larger toward the near side
      for (const [lat, lon] of LAND_POINTS) {
        const p = project(lat, lon, rot, R, cx, cy)
        if (!p) continue
        const alpha = 0.4 + p.depth * 0.6
        const rDot = 1.05 + p.depth * 1.35
        ctx.beginPath()
        ctx.arc(p.px, p.py, rDot, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(21, 87, 36, ${alpha.toFixed(2)})`
        ctx.fill()
      }

      // specular highlight + glass rim
      ctx.beginPath()
      ctx.ellipse(cx - R * 0.32, cy - R * 0.38, R * 0.32, R * 0.2, -0.5, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(255,255,255,0.14)'
      ctx.fill()
      ctx.beginPath()
      ctx.arc(cx, cy, R, 0, Math.PI * 2)
      ctx.lineWidth = 2
      ctx.strokeStyle = 'rgba(129, 199, 132, 0.5)'
      ctx.stroke()

      if (!reducedMotion) {
        angleRef.current += 0.0026
        raf = requestAnimationFrame(draw)
      }
    }
    draw()
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="globe"
      style={{ width: size, height: size }}
      role="img"
      aria-label="Rotating illustration of Earth"
    />
  )
}
