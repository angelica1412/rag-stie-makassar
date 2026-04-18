import React, { useState } from 'react';
import './MessageInput.css';

function MessageInput({ onSend, isLoading }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className="message-input-form" onSubmit={handleSubmit}>
      <div className="input-wrapper">
        <textarea
          className="message-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ketik pertanyaan Anda di sini..."
          rows={1}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-button"
          disabled={!input.trim() || isLoading}
        >
          {isLoading ? '⏳' : '➤'}
        </button>
      </div>
      <p className="input-hint">Tekan Enter untuk kirim, Shift+Enter untuk baris baru</p>
    </form>
  );
}

export default MessageInput;