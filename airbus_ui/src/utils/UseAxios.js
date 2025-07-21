import axios from 'axios';

const api = axios.create({
  // Add your base URL if needed
  // baseURL: 'your-api-base-url'
});

api.interceptors.request.use(config => {
  // Fixed: Use consistent token key
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  res => res,
  async error => {
    const originalRequest = error.config;
    
    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      localStorage.getItem('refresh_token')
    ) {
      originalRequest._retry = true;
      
      try {
        const refreshToken = localStorage.getItem('refresh_token');
        
        // Fixed: Use correct refresh endpoint and header format
        const res = await axios.post('/api/users/auth/refresh', null, {
          headers: {
            'refresh_token': refreshToken,
          },
        });
        
        const { access_token, refresh_token: newRefreshToken } = res.data;
        
        // Store both tokens
        localStorage.setItem('access_token', access_token);
        if (newRefreshToken) {
          localStorage.setItem('refresh_token', newRefreshToken);
        }
        
        // Update the default authorization header
        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        
        // Retry the original request
        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return api(originalRequest);
        
      } catch (err) {
        console.error('Token refresh failed:', err);
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/auth/login';
      }
    }
    
    return Promise.reject(error);
  }
);

export default api;