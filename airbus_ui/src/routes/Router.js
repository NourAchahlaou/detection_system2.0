import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router';

/* ***Layouts**** */
const FullLayout = lazy(() => import('../layouts/full/FullLayout'));
const BlankLayout = lazy(() => import('../layouts/blank/BlankLayout'));

/* ****Pages***** */
const Dashboard = lazy(() => import('../views/dashboard/Dashboard'))
const Clients = lazy(() => import('../views/airbus/Dashboard'))
const Router = [
  {
    path: '/',
    element: <FullLayout />,
    children: [
      { path: '/', element: <Navigate to="/dashboard" /> },
      { path: '/dashboard', exact: true, element: <Dashboard /> },
      { path : '/clients', element : <Clients />},
    ],
  },
  {
    path: '/auth',
    element: <BlankLayout />,
    // children: [
    //   { path: '404', element: <Error /> },
    //   { path: '/auth/register', element: <Register /> },
    //   { path: '/auth/login', element: <Login /> },
    //   { path: '*', element: <Navigate to="/auth/404" /> },
    // ],
  },
];

const router = createBrowserRouter(Router);

export default router;
