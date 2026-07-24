import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
})

export const sendMessage = (message, sessionId = 'default', location = null) =>
  api.post('/chat', {
    message,
    session_id: sessionId,
    location,
  })

export const predictFootprint = (data) =>
  api.post('/predict', data)

export const checkHealth = () =>
  api.get('/health')

// Get user's location via browser + reverse geocode
export const getUserLocation = () => {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      resolve(null)
      return
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const { latitude, longitude } = pos.coords
          // Use open-meteo's free geocoding (no API key needed)
          const resp = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`
          )
          const data = await resp.json()
          resolve({
            city:    data.address?.city || data.address?.town || data.address?.village || '',
            state:   data.address?.state || '',
            country: data.address?.country || '',
            country_code: data.address?.country_code?.toUpperCase() || '',
            lat: latitude,
            lon: longitude,
          })
        } catch {
          resolve(null)
        }
      },
      () => resolve(null),   // user denied — gracefully fallback
      { timeout: 5000 }
    )
  })
}

export default api