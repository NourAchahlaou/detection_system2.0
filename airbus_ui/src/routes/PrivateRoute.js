// src/components/PrivateRoute.jsx
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ children }) => {
  const { auth } = useAuth();
  return auth ? children : <Navigate to="/auth/login" />;
};

export default PrivateRoute;
