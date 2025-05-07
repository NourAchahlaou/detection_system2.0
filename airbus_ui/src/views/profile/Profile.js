import * as React from 'react';
import { useState } from 'react';
import { 
  Grid, 
  Box, 
  Paper, 
  Tabs, 
  Tab,
  Chip,
} from '@mui/material';
import { 
  Person as UserIcon, 
  CalendarToday as CalendarIcon, 
  PhotoCamera as CameraIcon, 
  Notifications as BellIcon, 
  BarChart as BarChartIcon, 
  Settings as SettingsIcon, 
} from '@mui/icons-material';

import ProfileTab from './ProfileTab';
import ShiftHistoryTab from './ShiftHistoryTab';
import ActivitiesTab from './ActivitiesTab';
import AlertsTab from './AlertsTab';
import PerformanceTab from './PerformanceTab';
import SettingsTab from './SettingsTab';
import SideCardProfile from './SideCardProfile';

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`profile-tabpanel-${index}`}
      aria-labelledby={`profile-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Main component
export default function MainDashboard() {
  const [activeTab, setActiveTab] = useState(0);
  
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}>
      <Grid container spacing={2}>
          
        {/* User Profile Side Card */}
        <Grid item xs={12} md={4} lg={3}>
          <SideCardProfile />
        </Grid>
        
        {/* Main Content Area */}
        <Grid item xs={12} md={8} lg={9}>
          <Box sx={{ width: '100%' }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange} 
              indicatorColor="primary"
              textColor="primary"
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab icon={<UserIcon fontSize="small" />} iconPosition="start" label="Profile" />
              <Tab icon={<CalendarIcon fontSize="small" />} iconPosition="start" label="Shift History" />
              <Tab icon={<CameraIcon fontSize="small" />} iconPosition="start" label="Activities" />
              <Tab 
                icon={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <BellIcon fontSize="small" />
                    <Chip
                      label="3"
                      color="error"
                      size="small"
                      sx={{ ml: 1, height: 20, fontSize: '0.75rem' }}
                    />
                  </Box>
                } 
                iconPosition="start" 
                label="Alerts" 
                sx={{ '& .MuiBadge-badge': { right: -3, top: 13 } }}
              />
              <Tab icon={<BarChartIcon fontSize="small" />} iconPosition="start" label="Performance" />
              <Tab icon={<SettingsIcon fontSize="small" />} iconPosition="start" label="Settings" />
            </Tabs>
            
            {/* Profile Tab Panel */}
            <TabPanel value={activeTab} index={0}>
              <ProfileTab />
            </TabPanel>
            
            {/* Shift History Tab Panel */}
            <TabPanel value={activeTab} index={1}>
              <ShiftHistoryTab />
            </TabPanel>
            
            {/* Activities Tab Panel */}
            <TabPanel value={activeTab} index={2}>
              <ActivitiesTab />
            </TabPanel>
            
            {/* Alerts Tab Panel */}
            <TabPanel value={activeTab} index={3}>
              <AlertsTab />
            </TabPanel>
            
            {/* Performance Tab Panel */}
            <TabPanel value={activeTab} index={4}>
              <PerformanceTab />
            </TabPanel>
            
            {/* Settings Tab Panel */}
            <TabPanel value={activeTab} index={5}>
              <SettingsTab />
            </TabPanel>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}