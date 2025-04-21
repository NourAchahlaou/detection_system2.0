// MenuContent.js
import React from 'react';
import List from '@mui/material/List';
import Stack from '@mui/material/Stack';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import AnalyticsRoundedIcon from '@mui/icons-material/AnalyticsRounded';
import PeopleRoundedIcon from '@mui/icons-material/PeopleRounded';
import AssignmentRoundedIcon from '@mui/icons-material/AssignmentRounded';
import SettingsRoundedIcon from '@mui/icons-material/SettingsRounded';
import InfoRoundedIcon from '@mui/icons-material/InfoRounded';
import HelpRoundedIcon from '@mui/icons-material/HelpRounded';
import SidebarItem from './SidebarItem';

const mainListItems = [
  { text: 'Home', icon: <HomeRoundedIcon />, to: '/' },
  { text: 'Analytics', icon: <AnalyticsRoundedIcon />, to: '/analytics' },
  { text: 'Clients', icon: <PeopleRoundedIcon />, to: '/clients' },
  { text: 'Tasks', icon: <AssignmentRoundedIcon />, to: '/tasks' },
];

const secondaryListItems = [
  { text: 'Settings', icon: <SettingsRoundedIcon />, to: '/settings' },
  { text: 'About', icon: <InfoRoundedIcon />, to: '/about' },
  { text: 'Feedback', icon: <HelpRoundedIcon />, to: '/feedback' },
];

const MenuContent = () => {
  return (
    <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'space-between' }}>
      <List dense>
        {mainListItems.map((item, index) => (
          <SidebarItem key={index} text={item.text} icon={item.icon} to={item.to} />
        ))}
      </List>
      <List dense>
        {secondaryListItems.map((item, index) => (
          <SidebarItem key={index} text={item.text} icon={item.icon} to={item.to} />
        ))} 
      </List>
    </Stack>
  );
};

export default MenuContent;
