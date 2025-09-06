// RoleBasedRoute.jsx - Enhanced with detailed debugging
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const RoleBasedRoute = ({ children, allowedRoles }) => {
  const { auth, loading } = useAuth();

  // Debug logging
  console.log('🔍 RoleBasedRoute Debug:', {
    loading,
    hasAuth: !!auth,
    hasUser: !!auth.user,
    userRole: auth.user?.role,
    userName: auth.user?.name,
    allowedRoles,
    timestamp: new Date().toISOString()
  });

  // Show loading state
  if (loading) {
    console.log('⏳ RoleBasedRoute: Still loading authentication...');
    return <div>Loading...</div>; // Or your loading component
  }

  // Check if user exists
  if (!auth.user) {
    console.log('❌ RoleBasedRoute: No user found, redirecting to login');
    return <Navigate to="/auth/login" replace />;
  }

  // Check if user has required role
  const hasPermission = () => {
    if (!auth.user || !auth.user.role) {
      console.log('❌ RoleBasedRoute: User or role missing', {
        hasUser: !!auth.user,
        userRole: auth.user?.role
      });
      return false;
    }

    const userRole = auth.user.role;
    const hasAccess = allowedRoles.includes(userRole);
    
    console.log('🔍 RoleBasedRoute Permission Check:', {
      userRole,
      allowedRoles,
      hasAccess
    });

    return hasAccess;
  };

  if (!hasPermission()) {
    console.log('🚫 RoleBasedRoute: Access denied, redirecting to dashboard');
    return <Navigate to="/dashboard" replace />;
  }

  console.log('✅ RoleBasedRoute: Access granted');
  return children;
};

export default RoleBasedRoute;