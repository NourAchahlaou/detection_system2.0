// MenuContent.js
import React from 'react';
import List from '@mui/material/List';
import Stack from '@mui/material/Stack';
import ListSubheader from '@mui/material/ListSubheader';

import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import CameraAltRoundedIcon from '@mui/icons-material/CameraAltRounded';
import LabelRoundedIcon from '@mui/icons-material/LabelRounded';
import FolderRoundedIcon from '@mui/icons-material/FolderRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import FindInPageRoundedIcon from '@mui/icons-material/FindInPageRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import DescriptionRoundedIcon from '@mui/icons-material/DescriptionRounded';
import AdminPanelSettingsRoundedIcon from '@mui/icons-material/AdminPanelSettingsRounded';
import BugReportRoundedIcon from '@mui/icons-material/BugReportRounded';
import ModelTrainingRoundedIcon from '@mui/icons-material/ModelTrainingRounded';

import SidebarItem from './SidebarItem';

const menuSections = [
  {
    title: 'Main',
    items: [
      { text: 'Dashboard', icon: <HomeRoundedIcon />, to: '/' },
    ],
  },
  {
    title: 'Image Operations',
    items: [
      { text: 'Capture Image', icon: <CameraAltRoundedIcon />, to: '/captureImage' },
      { text: 'Pieces Overview', icon: <LabelRoundedIcon />, to: '/piecesOverview' },
      { text: 'Manage Dataset', icon: <FolderRoundedIcon />, to: '/dataset' },
    ],
  },
  {
    title: 'Inspection & Identification',
    items: [
      { text: 'Verify Lot', icon: <SearchRoundedIcon />, to: '/detection' },
      { text: 'Identify Piece', icon: <FindInPageRoundedIcon />, to: '/identify' },
      { text: 'Inspection History', icon: <HistoryRoundedIcon />, to: '/history' },
    ],
  },
  {
    title: 'Reports & Traceability',
    items: [
      { text: 'Generate Reports', icon: <DescriptionRoundedIcon />, to: '/reports' },
      { text: 'Traceability Logs', icon: <HistoryRoundedIcon />, to: '/traceability' },
    ],
  },

  {
    title: 'Admin Panel',
    items: [
      { text: 'User Management', icon: <AdminPanelSettingsRoundedIcon />, to: '/admin/users' },
      { text: 'System Logs', icon: <BugReportRoundedIcon />, to: '/admin/logs' },
      { text: 'Model Management', icon: <ModelTrainingRoundedIcon />, to: '/admin/models' },
    ],
  },
];

const MenuContent = () => {
  return (
    <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'space-between' }}>
      {menuSections.map((section, idx) => (
        <List
          key={idx}
          dense
          subheader={
            <ListSubheader component="div" sx={{ bgcolor: 'inherit' }}>
              {section.title}
            </ListSubheader>
          }
        >
          {section.items.map((item, index) => (
            <SidebarItem
              key={index}
              text={item.text}
              icon={item.icon}
              to={item.to}
            />
          ))}
        </List>
      ))}
    </Stack>
  );
};

export default MenuContent;
