import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import api from '../utils/UseAxios';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState({ token: null, user: null });

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setAuth({ token: null, user: null });
  }, []);

  const fetchUser = useCallback(async () => {
    try {
      const res = await api.get('/users/me');
      setAuth((prev) => ({ ...prev, user: res.data }));
    } catch (error) {
      console.error('User fetch failed', error);
      logout();
    }
  }, [logout]);

  const refreshAccessToken = useCallback(async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      logout();
      return;
    }

    try {
      const res = await api.post('/auth/refresh', null, {
        headers: {
          refresh_token: refreshToken,
        },
      });

      const { access_token, refresh_token: newRefreshToken } = res.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', newRefreshToken);

      setAuth({ token: access_token, user: null }); // We'll get user again
      await fetchUser();
    } catch (error) {
      console.error('Token refresh failed', error);
      logout();
    }
  }, [fetchUser, logout]);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      setAuth((prev) => ({ ...prev, token }));
      fetchUser();
    }
  }, [fetchUser]);

  const login = async (token, refreshToken) => {
    localStorage.setItem('access_token', token);
    localStorage.setItem('refresh_token', refreshToken);
    setAuth({ token, user: null });
    await fetchUser();
  };

  return (
    <AuthContext.Provider value={{ auth, login, logout, refreshAccessToken }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
