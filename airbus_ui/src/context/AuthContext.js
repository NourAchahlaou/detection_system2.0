// AuthContext.jsx - Fixed with proper role handling and debugging
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
        console.log('🔍 No token found in localStorage');
        setLoading(false);
        return;
      }

      console.log('🔍 Fetching user data...');
      
      // Try the profile endpoint first (since it works in ProfileService)
      let userData = null;
      
      try {
        console.log('🔍 Trying /api/users/profile/ endpoint...');
        const profileRes = await api.get('/api/users/profile/');
        userData = profileRes.data;
        console.log('✅ Successfully fetched from /api/users/profile/:', userData);
      } catch (profileError) {
        console.log('❌ /api/users/profile/ failed, trying /api/users/users/me...');
        
        try {
          const userRes = await api.get('/api/users/users/me');
          userData = userRes.data;
          console.log('✅ Successfully fetched from /api/users/users/me:', userData);
        } catch (userError) {
          console.log('❌ /api/users/users/me failed, trying /api/users/profile/basic...');
          
          const basicRes = await api.get('/api/users/profile/basic');
          userData = basicRes.data;
          console.log('✅ Successfully fetched from /api/users/profile/basic:', userData);
        }
      }

      if (userData) {
        // Log user data for debugging
        console.log('🔍 User data received:', {
          id: userData.id,
          name: userData.name,
          email: userData.email,
          role: userData.role,
          airbus_id: userData.airbus_id,
          is_active: userData.is_active
        });
        
        // Check if role exists
        if (!userData.role) {
          console.warn('⚠️ WARNING: User role is missing or empty!', userData);
        } else {
          console.log('✅ User role found:', userData.role);
        }
        
        setAuth({ token, user: userData });
      } else {
        throw new Error('No user data received from any endpoint');
      }
      
    } catch (error) {
      console.error('❌ All user fetch attempts failed:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status
      });
      logout();
    } finally {
      setLoading(false);
    }
  }, [logout]);

  const refreshAccessToken = useCallback(async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      console.log('🔍 No refresh token found');
      logout();
      setLoading(false);
      return;
    }

    try {
      console.log('🔄 Refreshing access token...');
      const res = await api.post('/api/users/auth/refresh', null, {
        headers: {
          'refresh_token': refreshToken,
        },
      });

      const { access_token, refresh_token: newRefreshToken } = res.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', newRefreshToken);

      console.log('✅ Token refreshed successfully');
      setAuth(prev => ({ ...prev, token: access_token }));
      await fetchUser();
    } catch (error) {
      console.error('❌ Token refresh failed:', error);
      logout();
      setLoading(false);
    }
  }, [fetchUser, logout]);

  useEffect(() => {
    const initAuth = async () => {
      console.log('🚀 Initializing authentication...');
      setLoading(true);
      const token = localStorage.getItem('access_token');
      if (token) {
        console.log('🔍 Token found, fetching user...');
        setAuth(prev => ({ ...prev, token }));
        await fetchUser();
      } else {
        console.log('🔍 No token found, user not authenticated');
        setLoading(false);
      }
    };

    initAuth();
  }, [fetchUser]);

  const login = async (token, refreshToken) => {
    console.log('🔐 Logging in user...');
    localStorage.setItem('access_token', token);
    localStorage.setItem('refresh_token', refreshToken);
    setAuth({ token, user: null });
    await fetchUser();
  };

  // Debug function to log current auth state
  const debugAuthState = () => {
    console.log('🔍 Current Auth State:', {
      hasToken: !!auth.token,
      hasUser: !!auth.user,
      userRole: auth.user?.role,
      userName: auth.user?.name,
      loading
    });
  };

  return (
    <AuthContext.Provider value={{
      auth,
      login,
      logout,
      refreshAccessToken,
      loading,
      debugAuthState // Add debug function for testing
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);