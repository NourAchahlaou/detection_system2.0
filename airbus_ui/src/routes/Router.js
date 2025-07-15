import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';

import Loadable from "../components/components/load/loadable";

const SignUp = Loadable(lazy(()=> import( '../views/auth/sign-up/SignUp')));
const Profile = Loadable(lazy(()=> import( '../views/auth/profile-completion/Profile')));
const UserProfile = Loadable(lazy(()=> import( '../views/profile/Profile')));

/* Layouts */
const FullLayout = lazy(() => import('../layouts/full/FullLayout'));
const BlankLayout = lazy(() => import('../layouts/blank/BlankLayout'));

/* Pages */
const Dashboard = Loadable(lazy(() => import('../views/dashboard/Dashboard')));
const CaptureImage = Loadable(lazy(() => import('../views/captureImage/CameraCapture')));
const SignInSide = Loadable(lazy(() => import('../views/auth/sign-in-side/SignInSide')));
const Annotation = Loadable(lazy (()=> import('../views/imageAnnotaion/AppImageAnnotaion')));
const PiecesOverview = Loadable(lazy(() => import('../views/imageAnnotaion/PiecesOverview')));
const Dataset = Loadable(lazy(() => import('../views/dataset/datasetTable/Dataset')));

const NoData = Loadable(lazy(() => import("../views/sessions/NoData")));
const NoDataAnnotation = Loadable(lazy(() => import("../views/sessions/NoData_annotation")));

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
      {
        path: 'annotation',
        element: <Annotation />,
      },
      {
        path: 'piecesOverview',
        element: <PiecesOverview />,    
      },
      {
        path: 'dataset',
        element: <Dataset />, 
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
  // { path: "/404", element: <NotFound /> },
  { path: "/204", element: <NoData /> },
  { path: "/204_annotation", element: <NoDataAnnotation/> },
]);

export default router;