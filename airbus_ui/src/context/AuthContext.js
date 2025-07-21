// 1. Fixed AuthContext.jsx
import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import api from '../utils/UseAxios';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState({ token: null, user: null });
  const [loading, setLoading] = useState(true);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setAuth({ token: null, user: null });
  }, []);

  const fetchUser = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setLoading(false);
        return;
      }

      // Fixed: Use correct endpoint path
      const res = await api.get('/api/users/users/me');
      setAuth({ token, user: res.data });
    } catch (error) {
      console.error('User fetch failed', error);
      logout();
    } finally {
      setLoading(false);
    }
  }, [logout]);

  const refreshAccessToken = useCallback(async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      logout();
      setLoading(false);
      return;
    }

    try {
      // Fixed: Use correct header format
      const res = await api.post('/api/users/auth/refresh', null, {
        headers: {
          'refresh_token': refreshToken,
        },
      });

      const { access_token, refresh_token: newRefreshToken } = res.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', newRefreshToken);

      setAuth(prev => ({ ...prev, token: access_token }));
      await fetchUser();
    } catch (error) {
      console.error('Token refresh failed', error);
      logout();
      setLoading(false);
    }
  }, [fetchUser, logout]);

  useEffect(() => {
    const initAuth = async () => {
      setLoading(true);
      const token = localStorage.getItem('access_token');
      if (token) {
        setAuth(prev => ({ ...prev, token }));
        await fetchUser();
      } else {
        setLoading(false);
      }
    };
        
    initAuth();
  }, [fetchUser]);

  const login = async (token, refreshToken) => {
    localStorage.setItem('access_token', token);
    localStorage.setItem('refresh_token', refreshToken);
    setAuth({ token, user: null });
    await fetchUser();
  };

  return (
    <AuthContext.Provider value={{
      auth,
      login,
      logout,
      refreshAccessToken,
      loading
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);