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
      { text: 'Annotation', icon: <LabelRoundedIcon />, to: '/piecesOverview' },
      { text: 'Manage Dataset', icon: <FolderRoundedIcon />, to: '/dataset' },
      { text: 'Group Overview', icon: <FolderRoundedIcon />, to: '/PiecesGroupOverview' },
    ],
  },
  {
    title: 'Inspection & Identification',
    items: [
      { text: 'Verify Lot', icon: <SearchRoundedIcon />, to: '/detectionLotsOverview' },
      { text: 'Identify Piece', icon: <FindInPageRoundedIcon />, to: '/identification' },
      { text: 'Lot Session', icon: <HistoryRoundedIcon />, to: '/lotSessionViewer' },
    ],
  },
  
];

const MenuContent = () => {
  return (
    <Stack sx={{ flexGrow: 1, p: 1}}>
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
