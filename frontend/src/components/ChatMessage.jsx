import { Leaf, User } from 'lucide-react'
import './ChatMessage.css'

export default function ChatMessage({ role, content }) {
  const isUser = role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="avatar">
        {isUser ? <User size={16} /> : <Leaf size={16} color="#4caf50" />}
      </div>
      <div className={`bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {content.split('\n').map((line, i) => (
          <p key={i} style={{ margin: line === '' ? '6px 0' : '0' }}>{line}</p>
        ))}
      </div>
    </div>
  )
}