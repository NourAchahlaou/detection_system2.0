// Router.js
import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import PrivateRoute from './PrivateRoute';

/* Layouts */
const FullLayout = lazy(() => import('../layouts/full/FullLayout'));
const BlankLayout = lazy(() => import('../layouts/blank/BlankLayout'));

/* Pages */
const Dashboard = lazy(() => import('../views/dashboard/Dashboard'));
const CaptureImage = lazy(() => import('../views/captureImage/CaptureImage'));
const SignInSide = lazy(() => import('../views/auth/sign-in-side/SignInSide'));

const router = createBrowserRouter([
  {
    path: '/',
    element: <FullLayout/>,
    children: [
      { path: '/', element: <Navigate to="/dashboard" /> },
      {
        path: '/dashboard',
        element: (
          <PrivateRoute>
            <Dashboard />
          </PrivateRoute>
        ),
      },
      {
        path: '/captureImage',
        element: (
          <PrivateRoute>
            <CaptureImage />
          </PrivateRoute>
        ),
      },
    ],
  },
  {
    path: '/auth',
    element: <BlankLayout />,
    children: [
      { path: 'login', element: <SignInSide /> },
    ],
  },
]);

export default router;
