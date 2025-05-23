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
  alpha
} from '@mui/material';
import { 
  Edit as EditIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

export default function ProfileTab() {
  // Generate sample contribution data for the graph
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
  
  // Get color based on contribution value
  const getContributionColor = (value, theme) => {
    if (value === 0) return theme.palette.grey[100];
    if (value === 1) return alpha(theme.palette.primary.main, 0.2);
    if (value === 2) return alpha(theme.palette.primary.main, 0.4);
    if (value === 3) return alpha(theme.palette.primary.main, 0.7);
    return theme.palette.primary.main;
  };
  
  return (
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
          <Button 
            variant="text" 
            color="primary" 
            size="small"
            startIcon={<EditIcon fontSize="small" />}
          >
            Edit Profile
          </Button>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">First Name</Typography>
            <Typography variant="body2" fontWeight="medium">Achahlaou</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Last Name</Typography>
            <Typography variant="body2" fontWeight="medium">Nour</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Email</Typography>
            <Typography variant="body2" fontWeight="medium" noWrap>nour.achahlaou@airbus.com</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Role</Typography>
            <Typography variant="body2" fontWeight="medium">Technician</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Shift Start</Typography>
            <Typography variant="body2" fontWeight="medium">08:00</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Shift End</Typography>
            <Typography variant="body2" fontWeight="medium">16:00</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Current Status</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box sx={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                bgcolor: 'success.main', 
                mr: 1 
              }} />
              <Typography variant="body2" fontWeight="medium" color="success.main">Online</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Total Hours This Week</Typography>
            <Typography variant="body2" fontWeight="medium">31h 15min</Typography>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" fontWeight="medium">Access & Credentials</Typography>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Employee ID</Typography>
            <Typography variant="body2" fontWeight="medium">T-49276</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Work Area</Typography>
            <Typography variant="body2" fontWeight="medium">Assembly Line B</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Badge Number</Typography>
            <Typography variant="body2" fontWeight="medium">AIR-8541</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Access Level</Typography>
            <Typography variant="body2" fontWeight="medium">Level 2 (Standard Technician)</Typography>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" fontWeight="medium">{totalContributions} contributions in the last year</Typography>
          <Button 
            variant="text" 
            color="primary"
            size="small" 
            startIcon={<RefreshIcon fontSize="small" />}
          >
            Refresh
          </Button>
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
          </Box>
          
          <Box sx={{ 
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
            ))}
          </Box>
        </Card>
      </CardContent>
    </Card>
  );
}