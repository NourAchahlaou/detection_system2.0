// src/components/AuthWrapper.jsx
import { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';

const AuthWrapper = ({ children }) => {
  const { loading } = useAuth();
  const [isReady, setIsReady] = useState(false);
  
  useEffect(() => {
    if (!loading) {
      setIsReady(true);
    }
  }, [loading]);
  
  if (!isReady) {
    return <div>Initialisation de l'application...</div>; // Ou un spinner
  }
  
  return children;
};

export default AuthWrapper;