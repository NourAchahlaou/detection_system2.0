import axios from 'axios';

const api = axios.create({
});

api.interceptors.request.use(config => {
  const token = localStorage.getItem('accessToken');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  res => res,
  async error => {
    const originalRequest = error.config;
    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      localStorage.getItem('refreshToken')
    ) {
      originalRequest._retry = true;
      try {
        // Update the refresh endpoint to use nginx routing
        const res = await axios.post('/api/users/auth/refresh', {
          refresh_token: localStorage.getItem('refreshToken'),
        });
        const { access_token } = res.data;
        localStorage.setItem('accessToken', access_token);
        
        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        return api(originalRequest);
      } catch (err) {
        localStorage.clear();
        window.location.href = '/auth/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;