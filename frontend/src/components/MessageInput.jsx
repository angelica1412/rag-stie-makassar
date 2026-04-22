import React, { useState } from 'react';
import './MessageInput.css';

// Cek apakah input bermakna
function isValidQuestion(text) {
  const trimmed = text.trim();

  // Terlalu pendek
  if (trimmed.length < 5) return {
    valid: false,
    reason: 'Pertanyaan terlalu pendek. Minimal 5 karakter.'
  };

  // Hanya simbol/angka/karakter tidak bermakna
  const onlySymbolsOrNumbers = /^[\d\s\+\-\*\/\=\?\!\@\#\$\%\^\&\(\)\[\]\{\}\<\>\|\~\`\.,:;'"]+$/.test(trimmed);
  if (onlySymbolsOrNumbers) return {
    valid: false,
    reason: 'Pertanyaan tidak dapat dipahami. Silakan ketik pertanyaan dalam Bahasa Indonesia.'
  };

  // Hanya satu kata tanpa konteks
  const words = trimmed.split(/\s+/);
  if (words.length < 2) return {
    valid: false,
    reason: 'Pertanyaan terlalu singkat. Mohon berikan pertanyaan yang lebih lengkap.'
  };

  // Mengandung ekspresi matematika murni
  const isMath = /^[\d\s\+\-\*\/\=\(\)\.]+$/.test(trimmed);
  if (isMath) return {
    valid: false,
    reason: 'Sistem ini hanya menjawab pertanyaan seputar dokumen internal STIE Ciputra Makassar.'
  };

  return { valid: true, reason: null };
}

function MessageInput({ onSend, isLoading }) {
  const [input, setInput] = useState('');
  const [validationError, setValidationError] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const validation = isValidQuestion(input);
    if (!validation.valid) {
      setValidationError(validation.reason);
      setTimeout(() => setValidationError(null), 3000);
      return;
    }

    setValidationError(null);
    onSend(input);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChange = (e) => {
    setInput(e.target.value);
    if (validationError) setValidationError(null);
  };

  return (
    <div className="message-input-wrapper">
      {validationError && (
        <div className="validation-error">
          ⚠️ {validationError}
        </div>
      )}
      <form className="message-input-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <textarea
            className="message-input"
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ketik pertanyaan seputar aturan dan pedoman kampus..."
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
        <p className="input-hint">
          Sistem ini menjawab pertanyaan seputar dokumen internal STIE Ciputra Makassar
        </p>
      </form>
    </div>
  );
}

export default MessageInput;