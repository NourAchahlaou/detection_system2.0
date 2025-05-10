// App.js
import { Suspense } from 'react';
import { RouterProvider } from 'react-router-dom';
import router from './routes/Router';
import { AuthProvider } from './context/AuthContext';
import AuthWrapper from './components/AuthWrapper';
import Loading from './components/components/load/loading';

function App() {
  return (
    <AuthProvider>
      <AuthWrapper>
        <Suspense fallback={<Loading />}>
          <RouterProvider router={router} />
        </Suspense>
      </AuthWrapper>
    </AuthProvider>
  );
}

export default App;