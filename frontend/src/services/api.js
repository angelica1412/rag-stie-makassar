import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Endpoint untuk sivitas akademika ─────────────────────────────────────────

export const sendQuestion = async (question, userId = null) => {
  const response = await api.post('/chat', {
    question,
    user_id: userId,
  });
  return response.data;
};

export const checkQuestionStatus = async (questionId) => {
  const response = await api.get(`/chat/status/${questionId}`);
  return response.data;
};

// ── Endpoint untuk admin QA ───────────────────────────────────────────────────

export const getPendingQuestions = async () => {
  const response = await api.get('/admin/questions');
  return response.data;
};

export const getAllQuestions = async () => {
  const response = await api.get('/admin/questions/all');
  return response.data;
};

export const submitAnswer = async (questionId, answer) => {
  const response = await api.post('/admin/answer', {
    question_id: questionId,
    answer,
  });
  return response.data;
};

export default api;