// MenuContent.jsx - Enhanced with role debugging
import React, { useEffect } from 'react';
import List from '@mui/material/List';
import Stack from '@mui/material/Stack';
import ListSubheader from '@mui/material/ListSubheader';
import { useAuth } from '../../../context/AuthContext';

import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import CameraAltRoundedIcon from '@mui/icons-material/CameraAltRounded';
import LabelRoundedIcon from '@mui/icons-material/LabelRounded';
import FolderRoundedIcon from '@mui/icons-material/FolderRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import FindInPageRoundedIcon from '@mui/icons-material/FindInPageRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';

import SidebarItem from './SidebarItem';

// Role definitions
const ROLES = {
  DATA_MANAGER: "data manager",
  OPERATOR: "operator", 
  AUDITOR: "auditor",
  ADMIN: "admin"
};

const MenuContent = () => {
  const { auth, loading, debugAuthState } = useAuth();
  const userRole = auth.user?.role;

  // Debug effect to log menu rendering
  useEffect(() => {
    console.log('🔍 MenuContent Render Debug:', {
      loading,
      hasAuth: !!auth,
      hasUser: !!auth.user,
      userRole,
      userName: auth.user?.name,
      timestamp: new Date().toISOString()
    });

    // Call debug function if available
    if (debugAuthState) {
      debugAuthState();
    }
  }, [auth, loading, userRole, debugAuthState]);

  // Helper function to check if user has access to menu item
  const hasAccess = (allowedRoles) => {
    const access = allowedRoles.includes(userRole);
    console.log('🔍 Menu Access Check:', {
      userRole,
      allowedRoles,
      hasAccess: access
    });
    return access;
  };

  const menuSections = [
    // {
    //   title: 'Main',
    //   items: [
    //     {
    //       text: 'Dashboard',
    //       icon: <HomeRoundedIcon />,
    //       to: '/',
    //       roles: [ROLES.DATA_MANAGER, ROLES.OPERATOR, ROLES.AUDITOR, ROLES.ADMIN]
    //     },
    //   ],
    // },
    {
      title: 'Training & Data Management',
      items: [
        {
          text: 'Capture Image',
          icon: <CameraAltRoundedIcon />,
          to: '/captureImage',
          roles: [ROLES.DATA_MANAGER, ROLES.ADMIN]
        },
        {
          text: 'Annotation',
          icon: <LabelRoundedIcon />,
          to: '/piecesOverview',
          roles: [ROLES.DATA_MANAGER, ROLES.ADMIN]
        },
        {
          text: 'Manage Dataset',
          icon: <FolderRoundedIcon />,
          to: '/dataset',
          roles: [ROLES.DATA_MANAGER, ROLES.ADMIN]
        },
        {
          text: 'Group Overview',
          icon: <FolderRoundedIcon />,
          to: '/piecesGroupOverview',
          roles: [ROLES.DATA_MANAGER, ROLES.ADMIN,ROLES.AUDITOR]
        },
      ],
    },
    {
      title: 'Inspection & Operations',
      items: [
        {
          text: 'Verify Lot',
          icon: <SearchRoundedIcon />,
          to: '/detectionLotsOverview',
          roles: [ROLES.OPERATOR, ROLES.ADMIN]
        },
        {
          text: 'Identify Piece',
          icon: <FindInPageRoundedIcon />,
          to: '/identification',
          roles: [ROLES.OPERATOR, ROLES.ADMIN]
        },
      ],
    },
    {
      title: 'Reports & History',
      items: [
        {
          text: 'Lot Session',
          icon: <HistoryRoundedIcon />,
          to: '/lotSessionViewer',
          roles: [ROLES.OPERATOR, ROLES.AUDITOR, ROLES.ADMIN]
        },
      ],
    },
  ];

  // Show loading state
  if (loading) {
    return (
      <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'center', alignItems: 'center' }}>
        <div>Loading menu...</div>
      </Stack>
    );
  }

  // Show message if no user
  if (!auth.user) {
    console.log('❌ MenuContent: No user found');
    return (
      <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'center', alignItems: 'center' }}>
        <div>Please log in to see menu</div>
      </Stack>
    );
  }

  // Show message if no role
  if (!userRole) {
    console.log('❌ MenuContent: User has no role assigned');
    return (
      <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'center', alignItems: 'center' }}>
        <div>No role assigned. Please contact administrator.</div>
      </Stack>
    );
  }

  console.log('🔍 Processing menu sections for role:', userRole);

  return (
    <Stack sx={{ flexGrow: 1, p: 1}}>
      {menuSections.map((section, idx) => {
        // Filter items based on user role
        const accessibleItems = section.items.filter(item => {
          const hasItemAccess = hasAccess(item.roles);
          console.log(`🔍 Item "${item.text}" access:`, hasItemAccess);
          return hasItemAccess;
        });

        console.log(`🔍 Section "${section.title}" accessible items:`, accessibleItems.length);

        // Only show section if user has access to at least one item
        if (accessibleItems.length === 0) {
          console.log(`❌ Section "${section.title}" hidden - no accessible items`);
          return null;
        }

        console.log(`✅ Section "${section.title}" shown with ${accessibleItems.length} items`);

        return (
          <List
            key={idx}
            dense
            subheader={
              <ListSubheader component="div" sx={{ bgcolor: 'inherit' }}>
                {section.title}
              </ListSubheader>
            }
          >
            {accessibleItems.map((item, index) => (
              <SidebarItem
                key={index}
                text={item.text}
                icon={item.icon}
                to={item.to}
              />
            ))}
          </List>
        );
      })}
      
      {/* Debug info at bottom of menu */}
      <div style={{ 
        fontSize: '10px', 
        color: '#666', 
        padding: '8px', 
        borderTop: '1px solid #eee',
        marginTop: 'auto'
      }}>
        Role: {userRole} | User: {auth.user?.name}
      </div>
    </Stack>
  );
};

export default MenuContent;