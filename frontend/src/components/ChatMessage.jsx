import { Leaf, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatMessage.css'

// Links in agent replies open in a new tab so the chat isn't navigated away.
const markdownComponents = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />, // eslint-disable-line no-unused-vars
}

export default function ChatMessage({ role, content }) {
  const isUser = role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="avatar">
        {isUser ? <User size={16} /> : <Leaf size={16} color="#4caf50" />}
      </div>
      <div className={`bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {isUser ? (
          content.split('\n').map((line, i) => (
            <p key={i} style={{ margin: line === '' ? '6px 0' : '0' }}>{line}</p>
          ))
        ) : (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}