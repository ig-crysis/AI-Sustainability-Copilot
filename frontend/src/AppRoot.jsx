import { useEffect, useState } from 'react'
import App from './App.jsx'
import Landing from './components/Landing.jsx'
import { checkHealth } from './api'

export default function AppRoot() {
  const [started, setStarted] = useState(false)

  useEffect(() => {
    // Render's free tier cold-starts the backend; fire this the moment the
    // landing page mounts so it's warm by the time the user hits "Start".
    checkHealth().catch(() => {})
  }, [])

  return started ? <App /> : <Landing onStart={() => setStarted(true)} />
}
