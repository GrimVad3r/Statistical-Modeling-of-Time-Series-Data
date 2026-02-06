// src/App.tsx
import { useState, useEffect } from 'react';
import axios from 'axios';
import Dashboard from './components/Dashboard';  // ← adjust path if needed

function App() {
  const [apiHealthy, setApiHealthy] = useState(false);

  useEffect(() => {
    // Quick health check to backend (optional but useful)
    axios
      .get('http://localhost:5000/api/v1/health')
      .then(() => setApiHealthy(true))
      .catch((err) => console.error('Backend not reachable:', err));
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {apiHealthy ? (
        <Dashboard />
      ) : (
        <div className="p-8 text-center">
          <h1 className="text-3xl font-bold mb-4">Loading Dashboard...</h1>
          <p>Make sure your Flask backend is running on port 5000</p>
        </div>
      )}
    </div>
  );
}

export default App;