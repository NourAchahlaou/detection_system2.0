// utils/roleUtils.js - Utility functions for role-based permissions
export const ROLES = {
  DATA_MANAGER: "data manager",
  OPERATOR: "operator", 
  AUDITOR: "auditor"
};

// Define permissions for each role
export const ROLE_PERMISSIONS = {
  [ROLES.DATA_MANAGER]: {
    description: "Trainer - Can capture images, annotate datasets, and train AI models",
    permissions: [
      'capture_images',
      'annotate_data', 
      'manage_datasets',
      'view_pieces',
      'manage_pieces',
      'view_dashboard',
      'access_profile'
    ],
    restrictedActions: [
      'run_lot_checks',
      'manage_users'
    ]
  },
  
  [ROLES.OPERATOR]: {
    description: "Inspector - Can run lot conformity checks and identify pieces",
    permissions: [
      'run_lot_checks',
      'identify_pieces', 
      'verify_lots',
      'view_detection_results',
      'view_dashboard',
      'access_profile',
      'view_lot_sessions'
      
    ],
    restrictedActions: [
      'train_models',
      'add_new_pieces',
      'manage_datasets',
      'manage_users'
    ]
  },
  
  [ROLES.AUDITOR]: {
    description: "Viewer - Can view inspection results, reports, and statistics",
    permissions: [
      'view_reports',
      'view_statistics',
      'view_inspection_history',
      'view_dashboard',
      'access_profile'
    ],
    restrictedActions: [
      'modify_data',
      'run_operations', 
      'train_models',
      'manage_users'
    ]
  }
};

// Utility functions
export const hasPermission = (userRole, permission) => {
  if (!userRole || !ROLE_PERMISSIONS[userRole]) {
    return false;
  }
  return ROLE_PERMISSIONS[userRole].permissions.includes(permission);
};

export const hasAnyPermission = (userRole, permissions) => {
  return permissions.some(permission => hasPermission(userRole, permission));
};

export const canAccessRoute = (userRole, allowedRoles) => {
  if (!userRole || !allowedRoles) {
    return false;
  }
  return allowedRoles.includes(userRole);
};

export const getRoleDescription = (userRole) => {
  return ROLE_PERMISSIONS[userRole]?.description || "Unknown role";
};

export const getRolePermissions = (userRole) => {
  return ROLE_PERMISSIONS[userRole]?.permissions || [];
};

export const getRoleRestrictions = (userRole) => {
  return ROLE_PERMISSIONS[userRole]?.restrictedActions || [];
};