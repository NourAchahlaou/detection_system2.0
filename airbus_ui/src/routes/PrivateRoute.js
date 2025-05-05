// src/components/PrivateRoute.jsx
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ children }) => {
  const { auth, loading } = useAuth();
  const location = useLocation();
  
  // Attendre que le contexte d'authentification finisse de charger
  if (loading) {
    return <div>Chargement de l'authentification...</div>; // Ou un spinner
  }
  
  // Vérification de l'authentification une fois le chargement terminé
  const isAuthenticated = !!(auth.token && auth.user);
  
  if (!isAuthenticated) {
    return <Navigate to="/auth/login" state={{ from: location }} replace />;
  }
  
  return children;
};

export default PrivateRoute;