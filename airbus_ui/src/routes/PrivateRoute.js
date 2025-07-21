import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ children }) => {
  const { auth, loading } = useAuth();
  const location = useLocation();
  
  // Wait for authentication context to finish loading
  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
      }}>
        Loading authentication...
      </div>
    );
  }
  
  // Check authentication after loading is complete
  const isAuthenticated = !!(auth.token && auth.user);
  
  console.log('PrivateRoute - Auth state:', { 
    token: !!auth.token, 
    user: !!auth.user, 
    isAuthenticated 
  });
  
  if (!isAuthenticated) {
    return <Navigate to="/auth/login" state={{ from: location }} replace />;
  }
  
  return children;
};

export default PrivateRoute;