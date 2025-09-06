// Router.jsx - Updated with role-based access control
import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';

import Loadable from "../components/components/load/loadable";
import PrivateRoute from "./PrivateRoute";
import RoleBasedRoute from "../components/auth/RoleBasedRoute";

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
const Dataset = Loadable(lazy(() => import('../views/dataset/AppDatabasesetup')));
const Detection = Loadable(lazy(() => import('../views/detection/Detection')));
const NoData = Loadable(lazy(() => import("../views/sessions/NoData")));
const NoDataAnnotation = Loadable(lazy(() => import("../views/sessions/NoData_annotation")));
const DetectionLotsOverview = Loadable(lazy(() => import("../views/detection/DetectionLotsOverview")));
const Identification = Loadable(lazy(() => import("../views/identification/AppIdentification")));
const PiecesGroupOverview = Loadable(lazy(() => import("../views/pieces/PiecesGroupOverview")));
const PieceImageViewer = Loadable(lazy(() => import("../views/pieces/PieceImageViewer")));
const LotSessionViewer = Loadable(lazy(() => import("../views/lotSession/lotSessionDatabase")));

// Role definitions matching your enum
const ROLES = {
  DATA_MANAGER: "data manager",
  OPERATOR: "operator", 
  AUDITOR: "auditor"
};

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
      <PrivateRoute>
        <FullLayout />
      </PrivateRoute>  
    ),
    children: [
      {
        path: 'dashboard',
        element: <Dashboard />, // Accessible to all roles
      },
      {
        path: 'profile',
        element: <UserProfile />, // Accessible to all roles
      },
      
      // DATA MANAGER ONLY ROUTES
      {
        path: 'captureImage',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <CaptureImage />
          </RoleBasedRoute>
        ),
      },
      {
        path: 'annotation',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <Annotation />
          </RoleBasedRoute>
        ),
      },
      {
        path: 'piecesOverview',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <PiecesOverview />
          </RoleBasedRoute>
        ),    
      },
      {
        path: 'dataset',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <Dataset />
          </RoleBasedRoute>
        ), 
      },
      {
        path: 'piecesGroupOverview',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <PiecesGroupOverview />
          </RoleBasedRoute>
        ), 
      },
      {
        path: 'pieceImageViewer',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.DATA_MANAGER]}>
            <PieceImageViewer />
          </RoleBasedRoute>
        ),  
      },
      
      // OPERATOR ONLY ROUTES  
      {
        path: 'detectionLotsOverview',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.OPERATOR]}>
            <DetectionLotsOverview />
          </RoleBasedRoute>
        ),
      },
      {
        path: 'detection',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.OPERATOR]}>
            <Detection />
          </RoleBasedRoute>
        ),
      },
      {
        path: 'identification',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.OPERATOR]}>
            <Identification />
          </RoleBasedRoute>
        ),
      },
      
      // AUDITOR (VIEW ONLY) ROUTES
      {
        path: 'lotSessionViewer',
        element: (
          <RoleBasedRoute allowedRoles={[ROLES.AUDITOR, ROLES.OPERATOR]}>
            <LotSessionViewer />
          </RoleBasedRoute>
        ),
      }
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
  
  { path: "/204", element: <NoData /> },
  { path: "/204_annotation", element: <NoDataAnnotation/> },
]);

export default router;