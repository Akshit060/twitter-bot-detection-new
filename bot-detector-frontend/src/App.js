import React, { useState } from 'react';
import './App.css';

function App() {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) return;
    
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Twitter Bot Detector</h1>
        <p>Check if a Twitter account is a bot or human</p>
        
        <form onSubmit={handleSubmit} style={{ marginTop: '2rem', width: '100%', maxWidth: '400px' }}>
          <input
            type="text"
            placeholder="Enter Twitter username (without @)"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{
              width: '100%',
              padding: '0.75rem',
              fontSize: '1rem',
              borderRadius: '8px',
              border: '2px solid #61dafb',
              marginBottom: '1rem'
            }}
          />
          <button 
            type="submit" 
            disabled={loading}
            style={{
              width: '100%',
              padding: '0.75rem',
              fontSize: '1rem',
              backgroundColor: '#61dafb',
              color: '#282c34',
              border: 'none',
              borderRadius: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            {loading ? '🔍 Analyzing...' : '🔍 Check Account'}
          </button>
        </form>

        {error && (
          <div style={{
            marginTop: '2rem',
            padding: '1rem',
            backgroundColor: '#ff4444',
            borderRadius: '8px',
            color: 'white'
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div style={{
            marginTop: '2rem',
            padding: '1.5rem',
            backgroundColor: 'white',
            color: '#282c34',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            maxWidth: '400px',
            width: '100%'
          }}>
            <h2 style={{ margin: '0 0 1rem 0' }}>
              @{result.username}
            </h2>
            <div style={{ 
              fontSize: '3rem', 
              margin: '1rem 0',
              fontWeight: 'bold'
            }}>
              {result.prediction === 'BOT' ? '🤖 BOT' : '👤 HUMAN'}
            </div>
            <p style={{ fontSize: '1.2rem', margin: '0.5rem 0' }}>
              <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
            </p>
            <hr style={{ margin: '1rem 0', border: '1px solid #ddd' }} />
            <p style={{ margin: '0.5rem 0' }}>
              <strong>Bot Probability:</strong> {(result.bot_probability * 100).toFixed(1)}%
            </p>
            <p style={{ margin: '0.5rem 0' }}>
              <strong>Human Probability:</strong> {(result.human_probability * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
