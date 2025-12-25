import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

export const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="flex items-center space-x-2">
      <button
        onClick={() => setLanguage('en')}
        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200 ${
          language === 'en'
            ? 'bg-white text-blue-600 shadow-md'
            : 'bg-blue-500 text-white hover:bg-blue-400'
        }`}
      >
        🇬🇧 English
      </button>
      <button
        onClick={() => setLanguage('vi')}
        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200 ${
          language === 'vi'
            ? 'bg-white text-blue-600 shadow-md'
            : 'bg-blue-500 text-white hover:bg-blue-400'
        }`}
      >
        🇻🇳 Tiếng Việt
      </button>
    </div>
  );
};