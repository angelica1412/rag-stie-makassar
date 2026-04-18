import React, { useState, useEffect } from 'react';
import { getPendingQuestions, getAllQuestions, submitAnswer } from '../services/api';
import QuestionCard from '../components/QuestionCard';
import './AdminPage.css';

function AdminPage() {
  const [questions, setQuestions] = useState([]);
  const [activeTab, setActiveTab] = useState('pending');
  const [isLoading, setIsLoading] = useState(false);

  const fetchQuestions = async () => {
    setIsLoading(true);
    try {
      const data = activeTab === 'pending'
        ? await getPendingQuestions()
        : await getAllQuestions();
      setQuestions(data);
    } catch (err) {
      console.error('Error fetching questions:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
    // Auto refresh setiap 10 detik
    const interval = setInterval(fetchQuestions, 10000);
    return () => clearInterval(interval);
  }, [activeTab]);

  const handleAnswer = async (questionId, answer) => {
    try {
      await submitAnswer(questionId, answer);
      fetchQuestions();
    } catch (err) {
      console.error('Error submitting answer:', err);
    }
  };

  return (
    <div className="admin-page">
      <div className="admin-header">
        <h2>Dashboard Admin QA</h2>
        <p>Kelola pertanyaan yang memerlukan jawaban manual</p>
        <button className="refresh-btn" onClick={fetchQuestions}>
          🔄 Refresh
        </button>
      </div>

      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'pending' ? 'active' : ''}`}
          onClick={() => setActiveTab('pending')}
        >
          Menunggu Jawaban
          {questions.length > 0 && activeTab === 'pending' && (
            <span className="badge">{questions.length}</span>
          )}
        </button>
        <button
          className={`tab-btn ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          Semua Pertanyaan
        </button>
      </div>

      <div className="questions-list">
        {isLoading ? (
          <div className="loading">Memuat pertanyaan...</div>
        ) : questions.length === 0 ? (
          <div className="empty-state">
            <span>✅</span>
            <p>Tidak ada pertanyaan yang menunggu jawaban</p>
          </div>
        ) : (
          questions.map(q => (
            <QuestionCard
              key={q.question_id}
              question={q}
              onAnswer={handleAnswer}
              showAnswerForm={q.status === 'pending'}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default AdminPage;