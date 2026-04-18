import React from 'react';
import './ChatBubble.css';

function ChatBubble({ message }) {
  const isUser = message.type === 'user';

  return (
    <div className={`bubble-wrapper ${isUser ? 'user' : 'bot'}`}>
      {!isUser && (
        <div className="avatar">🤖</div>
      )}
      <div className={`bubble ${isUser ? 'user-bubble' : 'bot-bubble'} 
        ${message.isHitl ? 'hitl-bubble' : ''} 
        ${message.isError ? 'error-bubble' : ''}`}>
        <p>{message.text}</p>
        {message.sources && message.sources.length > 0 && (
          <div className="sources">
            <span className="sources-label">📄 Sumber:</span>
            <ul>
              {message.sources.map((src, idx) => (
                <li key={idx}>{src}</li>
              ))}
            </ul>
          </div>
        )}
        {message.isPending && (
          <div className="pending-indicator">
            ⏳ Menunggu jawaban dari staf QA...
          </div>
        )}
      </div>
      {isUser && (
        <div className="avatar user-avatar">👤</div>
      )}
    </div>
  );
}

export default ChatBubble;