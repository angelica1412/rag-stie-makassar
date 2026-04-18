import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import AdminPage from './pages/AdminPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="navbar-brand">
            <span className="brand-icon">🎓</span>
            <span className="brand-text">STIE Ciputra Makassar</span>
          </div>
          <div className="navbar-links">
            <Link to="/" className="nav-link">Chat</Link>
            <Link to="/admin" className="nav-link">Dashboard QA</Link>
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;