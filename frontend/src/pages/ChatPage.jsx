import React, { useState, useRef, useEffect } from 'react';
import { sendQuestion, checkQuestionStatus } from '../services/api';
import ChatBubble from '../components/ChatBubble';
import MessageInput from '../components/MessageInput';
import './ChatPage.css';

function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: 'Halo! Saya adalah asisten virtual STIE Ciputra Makassar. Silakan ajukan pertanyaan Anda mengenai dokumen internal kampus.',
      sources: [],
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingQuestionId, setPendingQuestionId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Polling untuk cek jawaban HITL
  useEffect(() => {
    if (!pendingQuestionId) return;

    const interval = setInterval(async () => {
      try {
        const status = await checkQuestionStatus(pendingQuestionId);
        if (status.status === 'answered' && status.answer) {
          setMessages(prev => [...prev, {
            id: Date.now(),
            type: 'bot',
            text: status.answer,
            sources: [],
            isHitl: true,
          }]);
          setPendingQuestionId(null);
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error checking status:', err);
      }
    }, 5000); // cek setiap 5 detik

    return () => clearInterval(interval);
  }, [pendingQuestionId]);

  const handleSend = async (question) => {
    if (!question.trim() || isLoading) return;

    // Tambahkan pesan pengguna
    setMessages(prev => [...prev, {
      id: Date.now(),
      type: 'user',
      text: question,
    }]);

    setIsLoading(true);

    try {
      const result = await sendQuestion(question);

      if (result.status === 'found') {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'bot',
          text: result.answer,
          sources: result.sources || [],
        }]);
      } else if (result.status === 'hitl_pending') {
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'bot',
          text: result.message,
          sources: [],
          isHitl: true,
          isPending: true,
        }]);
        setPendingQuestionId(result.question_id);
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'bot',
        text: 'Maaf, terjadi kesalahan. Silakan coba lagi.',
        sources: [],
        isError: true,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-header">
        <h2>Sistem Tanya Jawab Dokumen Internal</h2>
        <p>Ajukan pertanyaan seputar aturan dan prosedur kampus</p>
      </div>
      <div className="chat-messages">
        {messages.map(msg => (
          <ChatBubble key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="typing-indicator">
            <span></span><span></span><span></span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <MessageInput onSend={handleSend} isLoading={isLoading} />
    </div>
  );
}

export default ChatPage;