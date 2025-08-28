import * as React from 'react';
import { 
  Box, 
  Typography, 
  Avatar, 
  Chip, 
  Stack, 
  Divider,
  Card,
  CardContent,
  alpha,
  CircularProgress
} from '@mui/material';
import { 
  Schedule as ClockIcon, 
  Notifications as BellIcon, 
  PhotoCamera as CameraIcon, 
  Security as ShieldIcon,
  Email as EmailIcon,

} from '@mui/icons-material';
import profileService from './services/profileService'; // Adjust path as needed

export default function SideCard() {
  // State for profile data
  const [profileData, setProfileData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  // Fetch profile data
  const fetchProfileData = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await profileService.getProfileTabInfo();
      setProfileData(data);
    } catch (error) {
      console.error('Error fetching profile data for side card:', error);
      // You might want to show a toast notification here
    } finally {
      setLoading(false);
    }
  }, []);

  // Load profile data on component mount
  React.useEffect(() => {
    fetchProfileData();
  }, [fetchProfileData]);

  // Loading state
  if (loading) {
    return (
      <Card 
        elevation={2} 
        variant="outlined" 
        sx={{ 
          height: '100%',
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 2,
          overflow: 'auto',
          border: '1px solid',
          borderColor: 'divider',
          backgroundColor: (theme) => theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.8) : alpha(theme.palette.background.paper, 1),
        }}
      >
        <CircularProgress />
        <Typography variant="body2" sx={{ mt: 2 }} color="text.secondary">
          Loading profile...
        </Typography>
      </Card>
    );
  }

  // Error state or no data
  if (!profileData) {
    return (
      <Card 
        elevation={2} 
        variant="outlined" 
        sx={{ 
          height: '100%',
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 2,
          overflow: 'auto',
          border: '1px solid',
          borderColor: 'divider',
          backgroundColor: (theme) => theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.8) : alpha(theme.palette.background.paper, 1),
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Unable to load profile data
        </Typography>
      </Card>
    );
  }

  // Parse access level to extract level number and description
  const parseAccessLevel = (accessLevel) => {
    if (!accessLevel) return { level: 'N/A', description: 'No access level set' };
    
    // Example: "Level 2 (Standard Technician)" -> { level: "Level 2", description: "Standard Technician" }
    const match = accessLevel.match(/^(Level \d+)\s*\((.+)\)$/);
    if (match) {
      return { level: match[1], description: match[2] };
    }
    
    return { level: accessLevel, description: 'Access granted' };
  };

  // Parse work area to extract main area and section
  const parseWorkArea = (workArea) => {
    if (!workArea) return { main: 'No work area assigned', section: null };
    
    // For now, just return the work area as is
    // You can enhance this to parse more complex work area strings
    return { main: workArea}; // Keeping section static for now
  };

  const { level: accessLevel, description: accessDescription } = parseAccessLevel(profileData.access_level);
  const { main: workAreaMain } = parseWorkArea(profileData.work_area);
  const fullName = `${profileData.first_name || ''} ${profileData.last_name || ''}`.trim() || 'Unknown User';

  return (
    <Card 
      elevation={2} 
      variant="outlined" 
      sx={{ 
        height: '100%',
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'auto',
        border: '1px solid',
        borderColor: 'divider',
        backgroundColor: (theme) => theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.8) : alpha(theme.palette.background.paper, 1),
      }}
    >
      <CardContent sx={{ 
        p: 3,
        '&:last-child': { pb: 3 },
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>
        {/* Profile Header */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          width: '100%',
          position: 'relative',
          mb: 1
        }}>
          
          <Typography variant="h5" fontWeight="700" sx={{ mb: 0.5 }}>
            {fullName}
          </Typography>
          
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1.5, 
            my: 1,
            flexWrap: 'wrap',
            justifyContent: 'center'
          }}>
            <Chip 
              icon={<CameraIcon fontSize="small" />} 
              label={profileData.role || 'No role assigned'} 
              color="primary" 
              sx={{
                px: 1,
                fontWeight: 500,
                borderRadius: 1.5
              }}
              size="small"
            />
            
            <Chip
              sx={{ 
                bgcolor: (theme) => profileData.current_status === 'Online' 
                  ? alpha(theme.palette.success.main, 0.1)
                  : alpha(theme.palette.grey[500], 0.1),
                color: profileData.current_status === 'Online' ? 'success.dark' : 'text.secondary',
                fontWeight: 600,
                borderRadius: 1.5,
                '& .MuiChip-icon': {
                  color: profileData.current_status === 'Online' ? 'success.main' : 'grey.500'
                }
              }}
              icon={
                <Box component="span" sx={{ 
                  width: 8, 
                  height: 8, 
                  borderRadius: '50%', 
                  bgcolor: profileData.current_status === 'Online' ? 'success.main' : 'grey.500',
                  display: 'inline-block'
                }}/>
              }
              label={profileData.current_status === 'Online' ? 'Active Now' : 'Offline'}
              size="small"
            />
          </Box>
        </Box>
        
        <Divider sx={{ width: '100%' }} />
        
        {/* User Info Section */}
        <Box sx={{ width: '100%' }}>
          <Stack spacing={2} sx={{ fontSize: '0.875rem' }}>
            
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              pb: 1
            }}>
              <EmailIcon fontSize="small" sx={{ mr: 1.5, color: 'primary.main', fontSize: '1.1rem' }} />
              <Typography 
                variant="body2" 
                color="text.primary" 
                fontWeight={500} 
                noWrap 
                sx={{ 
                  maxWidth: '100%',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}
              >
                {profileData.email || 'No email available'}
              </Typography>
            </Box>
            
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              pb: 1
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <BellIcon fontSize="small" sx={{ mr: 1.5, color: 'primary.main', fontSize: '1.1rem' }} />
                <Typography variant="body2" color="text.secondary">
                {accessDescription}
              </Typography>
            </Box>
          </Box>
          </Stack>
        </Box>
        
        <Divider sx={{ width: '100%' }} />
        
        {/* Work Area Section */}
        <Box sx={{ width: '100%' }}>
          <Typography 
            variant="subtitle1" 
            fontWeight={600} 
            color="text.primary"
            sx={{ mb: 1.5 }}
          >
            Work Area
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            <Chip 
              label={workAreaMain} 
              variant="outlined" 
              size="medium" 
              color="primary"
              sx={{ 
                fontSize: '0.8rem',
                fontWeight: 500,
                borderRadius: 1.5,
                py: 0.5
              }}
            />
          </Box>
          
        </Box>
      </CardContent>
    </Card>
  );
}