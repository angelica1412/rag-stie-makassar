import React, { useState, useEffect, useRef } from 'react';
import { getPendingQuestions, getAllQuestions, submitAnswer } from '../services/api';
import QuestionCard from '../components/QuestionCard';
import './AdminPage.css';

function AdminPage() {
  const [questions, setQuestions] = useState([]);
  const [activeTab, setActiveTab]   = useState('pending');
  const [isLoading, setIsLoading]   = useState(false);
  const [answers, setAnswers]       = useState({});

  // Gunakan ref — tidak menyebabkan re-render saat berubah
  const hasActiveDraftRef = useRef(false);

  const handleAnswerChange = (questionId, value) => {
    setAnswers(prev => {
      const updated = { ...prev, [questionId]: value };
      // Update ref berdasarkan apakah ada draft yang diketik
      hasActiveDraftRef.current = Object.values(updated).some(
        v => v && v.trim().length > 0
      );
      return updated;
    });
  };

  const fetchQuestions = async () => {
    // Cek via ref — tidak trigger re-render
    if (hasActiveDraftRef.current) {
      console.log('[ADMIN] Skip refresh — ada draft yang belum dikirim');
      return;
    }

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
    const interval = setInterval(fetchQuestions, 10000);
    return () => clearInterval(interval);
  }, [activeTab]);

  const handleAnswer = async (questionId) => {
    const answer = answers[questionId];
    if (!answer || !answer.trim()) return;

    try {
      await submitAnswer(questionId, answer);
      setAnswers(prev => {
        const updated = { ...prev };
        delete updated[questionId];
        // Update ref setelah draft dihapus
        hasActiveDraftRef.current = Object.values(updated).some(
          v => v && v.trim().length > 0
        );
        return updated;
      });
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
        <button className="refresh-btn" onClick={() => {
          setAnswers({});
          hasActiveDraftRef.current = false;
          fetchQuestions();
        }}>
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
              answerValue={answers[q.question_id] || ''}
              onAnswerChange={(value) => handleAnswerChange(q.question_id, value)}
              onAnswer={() => handleAnswer(q.question_id)}
              showAnswerForm={q.status === 'pending'}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default AdminPage;