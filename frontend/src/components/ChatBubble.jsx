import React, { useState } from "react";
import PreviewModal from "./PreviewModal";
import "./ChatBubble.css";

function ChatBubble({ message }) {
  const isUser = message.type === "user";
  const [previewFile, setPreviewFile] = useState(null);

  return (
    <div className={`bubble-wrapper ${isUser ? "user" : "bot"}`}>
      {!isUser && <div className="avatar">🤖</div>}

      <div
        className={`bubble ${isUser ? "user-bubble" : "bot-bubble"}
        ${message.isHitl ? "hitl-bubble" : ""}
        ${message.isNotRelevant ? "not-relevant-bubble" : ""}
        ${message.isError ? "error-bubble" : ""}`}
      >
        <div className="bubble-text">
          {message.text.split("\n").map((line, idx) => {
            // Deteksi baris kosong
            if (line.trim() === "") return <br key={idx} />;

            // Deteksi numbered list (1. 2. 3.)
            const numberedMatch = line.match(/^(\d+)\.\s+(.*)/);
            if (numberedMatch) {
              return (
                <div key={idx} className="list-item">
                  <span className="list-number">{numberedMatch[1]}.</span>
                  <span>{numberedMatch[2]}</span>
                </div>
              );
            }

            // Deteksi bullet points (- atau •)
            const bulletMatch = line.match(/^[-•]\s+(.*)/);
            if (bulletMatch) {
              return (
                <div key={idx} className="list-item">
                  <span className="list-bullet">•</span>
                  <span>{bulletMatch[1]}</span>
                </div>
              );
            }

            // Teks biasa
            return (
              <p key={idx} className="bubble-line">
                {line}
              </p>
            );
          })}
        </div>

        {/* Tampilan khusus untuk form */}
        {message.isFormResponse && message.sources && (
          <div className="form-list">
            {message.sources.map((src, idx) => {
              const filename = typeof src === "object" ? src.filename : src;
              const label = typeof src === "object" ? src.label : src;
              const isPdf = filename && filename.toLowerCase().endsWith(".pdf");

              return (
                <div key={idx} className="form-item">
                  <span className="form-icon">📄</span>
                  <span className="form-name">{label}</span>
                  <div className="form-actions">
                    {isPdf && (
                      <button
                        className="btn-preview"
                        onClick={() =>
                          setPreviewFile({ name: label, filename })
                        }
                      >
                        👁️ Preview
                      </button>
                    )}
                    <a
                      href={`http://localhost:8000/download/${filename}`}
                      download={filename}
                      className="btn-download-small"
                    >
                      ⬇️ Download
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Tampilan sumber untuk jawaban naratif */}
        {!message.isFormResponse &&
          message.sources &&
          message.sources.length > 0 && (
            <div className="sources">
              <span className="sources-label">📄 Sumber dokumen:</span>
              <ul>
                {message.sources.map((src, idx) => (
                  <li key={idx}>{typeof src === "object" ? src.label : src}</li>
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

      {isUser && <div className="avatar user-avatar">👤</div>}

      {previewFile && (
        <PreviewModal file={previewFile} onClose={() => setPreviewFile(null)} />
      )}
    </div>
  );
}

export default ChatBubble;
