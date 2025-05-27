import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';

const SignUp = lazy(()=> import( '../views/auth/sign-up/SignUp'));
const Profile = lazy(()=> import( '../views/auth/profile-completion/Profile'));
const UserProfile = lazy(()=> import( '../views/profile/Profile'));

/* Layouts */
const FullLayout = lazy(() => import('../layouts/full/FullLayout'));
const BlankLayout = lazy(() => import('../layouts/blank/BlankLayout'));

/* Pages */
const Dashboard = lazy(() => import('../views/dashboard/Dashboard'));
const CaptureImage = lazy(() => import('../views/captureImage/CameraCapture'));
const SignInSide = lazy(() => import('../views/auth/sign-in-side/SignInSide'));

const router = createBrowserRouter([
  // Root redirect to dashboard
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },

  // Protected layout and routes
  {
    path: '/',
    element: (
      <FullLayout />
      // {/* <PrivateRoute>
      //   <FullLayout />
      // </PrivateRoute> */}
    ),
    children: [
      {
        path: 'dashboard',
        element: <Dashboard />,
      },
      {
        path: 'captureImage',
        element: <CaptureImage />,
      },
      {
        path: 'profile',
        element: <UserProfile />,
      },
    ],
  },

  // Public login route
  {
    path: '/auth',
    element: <BlankLayout />,
    children: [
      {
        path: 'login',
        element: <SignInSide />,
      },
      {
        path: 'signup',
        element: <SignUp />,
      },
      {
        path: 'profile',
        element: <Profile />,
      },

    ],
  },
]);

export default router;