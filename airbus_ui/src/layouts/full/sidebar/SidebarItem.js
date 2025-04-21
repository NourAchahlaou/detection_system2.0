// SidebarItem.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ListItem, ListItemButton, ListItemIcon, ListItemText } from '@mui/material';

// SidebarItem.js
const SidebarItem = ({ text, icon, to, handleClick }) => {
  const location = useLocation();

  return (
    <ListItem disablePadding sx={{ display: 'block' }}>
      <ListItemButton
        component={Link}
        to={to}
        selected={location.pathname === to}
        onClick={handleClick}
      >
        <ListItemIcon>{icon}</ListItemIcon>
        <ListItemText primary={text} />
      </ListItemButton>
    </ListItem>
  );
};

export default SidebarItem;
