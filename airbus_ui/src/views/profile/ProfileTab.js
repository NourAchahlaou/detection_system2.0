import * as React from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Divider,
  Tooltip,
  alpha,
  CircularProgress,
  Snackbar,
  Alert,
  Chip,
  Stack,
  IconButton
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Analytics as AnalyticsIcon,
  Schedule as ScheduleIcon,
  TrendingUp as TrendingUpIcon,
  Weekend as WeekendIcon
} from '@mui/icons-material';
import { ProfileService } from './services/AccountService';
import ProfileUpdateDialog from './ProfileUpdateDialog';

// Initialize the service
const profileService = new ProfileService();

export default function ProfileTab() {
  // State for profile data
  const [profileData, setProfileData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [refreshing, setRefreshing] = React.useState(false);
  const [notification, setNotification] = React.useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // State for the update dialog
  const [updateDialogOpen, setUpdateDialogOpen] = React.useState(false);

  // Generate sample contribution data for the graph (keeping this as static for now)
  const generateContributionData = () => {
    const weeks = 52;
    const days = 7;
    const data = [];
   
    for (let i = 0; i < weeks; i++) {
      const week = [];
      for (let j = 0; j < days; j++) {
        // Random value between 0-4 for contribution intensity
        const value = Math.floor(Math.random() * 5);
        week.push(value);
      }
      data.push(week);
    }
   
    return data;
  };
 
  const contributionData = React.useMemo(() => generateContributionData(), []);
 
  // Calculate total contributions
  const totalContributions = React.useMemo(() => {
    return contributionData.flat().reduce((sum, value) => sum + value, 0);
  }, [contributionData]);

  // Show notification helper
  const showNotification = React.useCallback((message, severity) => {
    setNotification({
      open: true,
      message,
      severity
    });
  }, []);

  // Close notification handler
  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  // Fetch profile data using ProfileService
  const fetchProfileData = React.useCallback(async () => {
    try {
      setLoading(true);
      
      const result = await profileService.getUserProfile();
      
      if (result.success) {
        // Transform the backend data to match your existing component structure
        const transformedData = {
          name: result.profile.name || 'N/A',
          first_name: result.profile.name ? result.profile.name.split(' ')[0] : 'N/A',
          last_name: result.profile.name ? result.profile.name.split(' ').slice(1).join(' ') : 'N/A',
          email: result.profile.email || 'N/A',
          role: result.profile.role || 'N/A',
          airbus_id: result.profile.airbus_id || 'N/A',
          current_status: result.profile.is_active ? 'Online' : 'Offline',
          // Keep existing fields that don't have backend equivalents
          shift_start: 'N/A',
          shift_end: 'N/A',
          total_hours_this_week: '0h 0min',
          work_area: 'N/A',
          badge_number: 'N/A',
          access_level: 'Standard',
          // Add backend-specific data for potential future use
          _original: result.profile
        };
        
        setProfileData(transformedData);
      } else {
        throw new Error(result.message || 'Failed to load profile data');
      }
      
    } catch (error) {
      console.error('Error fetching profile data:', error);
      showNotification(error.message || 'Failed to load profile data', 'error');
    } finally {
      setLoading(false);
    }
  }, [showNotification]);

  // Refresh data handler
  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      
      const result = await profileService.refreshProfile();
      
      if (result.success) {
        // Transform the refreshed data
        const transformedData = {
          first_name: result.profile.name ? result.profile.name.split(' ')[0] : 'N/A',
          last_name: result.profile.name ? result.profile.name.split(' ').slice(1).join(' ') : 'N/A',
          email: result.profile.email || 'N/A',
          role: result.profile.role || 'N/A',
          airbus_id: result.profile.airbus_id || 'N/A',
          current_status: result.profile.is_active ? 'Online' : 'Offline',
          shift_start: 'N/A',
          shift_end: 'N/A',
          total_hours_this_week: '0h 0min',
          work_area: 'N/A',
          badge_number: 'N/A',
          access_level: 'Standard',
          _original: result.profile
        };
        
        setProfileData(transformedData);
        showNotification('Profile data refreshed successfully', 'success');
      } else {
        throw new Error(result.message || 'Failed to refresh profile data');
      }
      
    } catch (error) {
      console.error('Error refreshing profile data:', error);
      showNotification(error.message || 'Failed to refresh profile data', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  // Handle profile update from dialog
  const handleProfileUpdate = React.useCallback((updatedData) => {
    if (updatedData === null) {
      // Profile was deleted
      showNotification('Profile deleted successfully. Redirecting...', 'info');
      // You might want to redirect to login or home page here
      return;
    }

    if (updatedData) {
      // Transform the updated data to match component structure
      const transformedData = {
        first_name: updatedData.name ? updatedData.name.split(' ')[0] : 'N/A',
        last_name: updatedData.name ? updatedData.name.split(' ').slice(1).join(' ') : 'N/A',
        email: updatedData.email || 'N/A',
        role: updatedData.role || 'N/A',
        airbus_id: updatedData.airbus_id || 'N/A',
        current_status: updatedData.is_active ? 'Online' : 'Offline',
        shift_start: profileData?.shift_start || 'N/A',
        shift_end: profileData?.shift_end || 'N/A',
        total_hours_this_week: profileData?.total_hours_this_week || '0h 0min',
        work_area: profileData?.work_area || 'N/A',
        badge_number: profileData?.badge_number || 'N/A',
        access_level: profileData?.access_level || 'Standard',
        _original: updatedData
      };
      
      setProfileData(transformedData);
      showNotification('Profile updated successfully', 'success');
    }
    
    setUpdateDialogOpen(false);
  }, [showNotification, profileData]);

  // Open update dialog handler
  const handleOpenUpdateDialog = () => {
    setUpdateDialogOpen(true);
  };

  // Close update dialog handler
  const handleCloseUpdateDialog = () => {
    setUpdateDialogOpen(false);
  };

  // Load profile data on component mount
  React.useEffect(() => {
    fetchProfileData();
  }, [fetchProfileData]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      profileService.cleanup();
    };
  }, []);
 
  // Get color based on contribution value
  const getContributionColor = (value, theme) => {
    if (value === 0) return theme.palette.grey[100];
    if (value === 1) return alpha(theme.palette.primary.main, 0.2);
    if (value === 2) return alpha(theme.palette.primary.main, 0.4);
    if (value === 3) return alpha(theme.palette.primary.main, 0.7);
    return theme.palette.primary.main;
  };

  // Loading state
  if (loading) {
    return (
      <Card
        elevation={1}
        variant="outlined"
        sx={{
          borderRadius: 1,
          overflow: 'auto',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Loading profile information...
        </Typography>
      </Card>
    );
  }

  // Error state
  if (!profileData) {
    return (
      <Card
        elevation={1}
        variant="outlined"
        sx={{
          borderRadius: 1,
          overflow: 'auto',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <Typography variant="h6" color="error" gutterBottom>
          Failed to load profile data
        </Typography>
        <Button
          variant="outlined"
          onClick={fetchProfileData}
          startIcon={<RefreshIcon />}
        >
          Try Again
        </Button>
      </Card>
    );
  }
 
  return (
    <>
      <Card
        elevation={1}
        variant="outlined"
        sx={{
          borderRadius: 1,
          overflow: 'auto',
          height: '100%',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" fontWeight="medium">Personal Information</Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="text"
                color="primary"
                size="small"
                startIcon={<RefreshIcon fontSize="small" />}
                onClick={handleRefresh}
                disabled={refreshing}
              >
                {refreshing ? 'Refreshing...' : 'Refresh'}
              </Button>
              <Button
                variant="text"
                color="primary"
                size="small"
                startIcon={<EditIcon fontSize="small" />}
                onClick={handleOpenUpdateDialog}
              >
                Edit Profile
              </Button>
            </Box>
          </Box>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">First Name</Typography>
              <Typography variant="body2" fontWeight="medium">
                {profileData.first_name || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Last Name</Typography>
              <Typography variant="body2" fontWeight="medium">
                {profileData.last_name || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Email</Typography>
              <Typography variant="body2" fontWeight="medium" noWrap>
                {profileData.email || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Role</Typography>
              <Typography variant="body2" fontWeight="medium">
                {profileData.role || 'N/A'}
              </Typography>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Current Status</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ 
                  width: 8, 
                  height: 8, 
                  borderRadius: '50%', 
                  bgcolor: profileData.current_status === 'Online' ? 'success.main' : 'grey.500', 
                  mr: 1 
                }} />
                <Typography 
                  variant="body2" 
                  fontWeight="medium" 
                  color={profileData.current_status === 'Online' ? 'success.main' : 'text.secondary'}
                >
                  {profileData.current_status || 'Offline'}
                </Typography>
              </Box>
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" fontWeight="medium">Access & Credentials</Typography>
          </Box>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Work Area</Typography>
              <Typography variant="body2" fontWeight="medium">
                Quality Control
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Airbus ID</Typography>
              <Typography variant="body2" fontWeight="medium">
                {profileData.airbus_id || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="caption" color="text.secondary">Access Level</Typography>
              <Typography variant="body2" fontWeight="medium">
                {profileData.access_level || 'N/A'}
              </Typography>
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 2 }} />
{/*           
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" fontWeight="medium">{totalContributions} contributions in the last year</Typography>
          </Box>
          
          <Card
            variant="outlined"
            sx={{
              p: 2,
              bgcolor: 'background.paper',
            }}
          >
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              mb: 1.5,
              fontSize: '0.75rem'
            }}>
              <Typography variant="caption" color="text.secondary">May - Apr</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>Less</Typography>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    bgcolor: theme => theme.palette.grey[100],
                    border: '1px solid',
                    borderColor: 'divider'
                  }} />
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    bgcolor: theme => alpha(theme.palette.primary.main, 0.2),
                    border: '1px solid',
                    borderColor: 'divider'
                  }} />
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    bgcolor: theme => alpha(theme.palette.primary.main, 0.4),
                    border: '1px solid',
                    borderColor: 'divider'
                  }} />
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    bgcolor: theme => alpha(theme.palette.primary.main, 0.7),
                    border: '1px solid',
                    borderColor: 'divider'
                  }} />
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    bgcolor: 'primary.main',
                    border: '1px solid',
                    borderColor: 'divider'
                  }} />
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>More</Typography>
              </Box>
            </Box> */}
            
            {/* <Box sx={{ 
              display: 'flex', 
              flexWrap: 'wrap',
              gap: 0,
              maxWidth: '100%',
              overflowX: 'auto'
            }}>
              {contributionData.map((week, weekIndex) => (
                <Box 
                  key={`week-${weekIndex}`} 
                  sx={{ display: 'flex', flexDirection: 'column' }}
                >
                  {week.map((value, dayIndex) => (
                    <Tooltip 
                      key={`day-${weekIndex}-${dayIndex}`}
                      title={`${value} contribution${value !== 1 ? 's' : ''}`}
                      arrow
                    >
                      <Box sx={{ 
                        width: 10, 
                        height: 10, 
                        m: 0.25,
                        bgcolor: theme => getContributionColor(value, theme),
                        border: '1px solid',
                        borderColor: 'divider',
                        cursor: 'pointer',
                        '&:hover': {
                          boxShadow: 1
                        }
                      }} />
                    </Tooltip>
                  ))}
                </Box>
            //   ))} 
            // </Box>
          </Card>*/}
        </CardContent>
      </Card>

      {/* Profile Update Dialog */}
      <ProfileUpdateDialog
        open={updateDialogOpen}
        onClose={handleCloseUpdateDialog}
        onProfileUpdate={handleProfileUpdate}
        profileData={profileData}
      />

      {/* Notification Snackbar */}
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
}