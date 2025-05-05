// App.js
import { Suspense } from 'react';
import { RouterProvider } from 'react-router-dom';
import router from './routes/Router';
import { AuthProvider } from './context/AuthContext';
import AuthWrapper from './components/AuthWrapper';

function App() {
  return (
    <AuthProvider>
      <AuthWrapper>
        <Suspense fallback={<div>Chargement...</div>}>
          <RouterProvider router={router} />
        </Suspense>
      </AuthWrapper>
    </AuthProvider>
  );
}

export default App;