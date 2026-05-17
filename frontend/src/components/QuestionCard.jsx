import React, { useState } from 'react';
import './QuestionCard.css';

function QuestionCard({
  question,
  onAnswer,
  onAnswerChange, 
  answerValue, 
  showAnswerForm
}) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!answerValue.trim()) return;
    setIsSubmitting(true);
    await onAnswer(question.question_id, answerValue);
    setIsSubmitting(false);
  };

  const formatDate = (isoString) => {
    if (!isoString) return '-';
    return new Date(isoString).toLocaleString('id-ID');
  };

  return (
    <div className={`question-card ${question.status === 'answered' ? 'answered' : ''}`}>
      <div className="card-header">
        <span className={`status-badge ${question.status}`}>
          {question.status === 'pending' ? '⏳ Menunggu' : '✅ Terjawab'}
        </span>
        <span className="timestamp">{formatDate(question.timestamp)}</span>
      </div>

      <div className="question-text">
        <strong>Pertanyaan:</strong>
        <p>{question.question}</p>
      </div>

      {question.status === 'answered' && question.answer && (
        <div className="answer-text">
          <strong>Jawaban:</strong>
          <p>{question.answer}</p>
          <span className="answered-at">
            Dijawab: {formatDate(question.answered_at)}
          </span>
        </div>
      )}

      {showAnswerForm && (
        <div className="answer-form">
          <textarea
            value={answerValue}           // ← dari AdminPage
            onChange={(e) => onAnswerChange(e.target.value)}  // ← ke AdminPage
            placeholder="Tulis jawaban untuk pertanyaan ini..."
            rows={3}
          />
          <button
            onClick={handleSubmit}
            disabled={!answerValue.trim() || isSubmitting}
            className="submit-btn"
          >
            {isSubmitting ? 'Mengirim...' : 'Kirim Jawaban'}
          </button>
        </div>
      )}
    </div>
  );
}

export default QuestionCard;