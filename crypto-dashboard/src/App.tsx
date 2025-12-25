import React from 'react';
import { Dashboard } from './components/Dashboard';
import { LanguageProvider } from '../src/contexts/LanguageContext';
import './App.css';

function App() {
  return (
    <LanguageProvider>
      <div className="App">
        <Dashboard />
      </div>
    </LanguageProvider>
  );
}

export default App;