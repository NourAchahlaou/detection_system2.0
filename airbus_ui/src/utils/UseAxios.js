import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8001',
});

api.interceptors.request.use(config => {
  // Update this line to match your actual storage key
  const token = localStorage.getItem('accessToken'); // Changed from access_token to accessToken
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
        const res = await axios.post('http://localhost:8001/auth/refresh', {
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