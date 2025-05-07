import * as React from 'react';
import { 
  Paper, 
  Box, 
  Typography, 
  Avatar, 
  Chip, 
  Button, 
  Stack, 
  IconButton, 
  Divider 
} from '@mui/material';
import { 
  Schedule as ClockIcon, 
  Notifications as BellIcon, 
  PhotoCamera as CameraIcon, 
  Logout as LogOutIcon, 
  Security as ShieldIcon,
  Email as EmailIcon 
} from '@mui/icons-material';

export default function SideCard() {
  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 3, 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: { xs: 'center', md: 'flex-start' }
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
        <Avatar 
          src="/api/placeholder/400/400" 
          alt="User avatar" 
          sx={{ width: 120, height: 120, mb: 2 }}
        />
        <Typography variant="h5" fontWeight="bold">Achahlaou Nour</Typography>
        
        <Chip 
          icon={<CameraIcon fontSize="small" />} 
          label="Technician" 
          color="primary" 
          variant="outlined" 
          sx={{ my: 1 }} 
        />
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{ 
            width: 10, 
            height: 10, 
            borderRadius: '50%', 
            bgcolor: 'success.main', 
            mr: 1 
          }} />
          <Typography color="success.main" fontWeight="medium">Active</Typography>
        </Box>
      </Box>
      
      <Box sx={{ width: '100%', mt: 2 }}>
        <Stack spacing={1.5}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ClockIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary" mr={1}>Working Hours Today:</Typography>
            <Typography variant="body2" fontWeight="medium">07h 45min</Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <EmailIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">nour.achahlaou@airbus.com</Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ClockIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary" mr={1}>Last Login:</Typography>
            <Typography variant="body2" fontWeight="medium">07 May 2025, 08:15</Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <BellIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary" mr={1}>Alerts:</Typography>
            <Chip
              label="3"
              color="error"
              size="small"
              sx={{ height: 20, fontSize: '0.75rem' }}
            />
          </Box>
        </Stack>
      </Box>
      
      <Box sx={{ width: '100%', mt: 3, display: 'flex', gap: 1 }}>
        <Button 
          variant="contained" 
          startIcon={<ClockIcon />}
          fullWidth
          size="small"
        >
          Start Shift
        </Button>
        <IconButton color="inherit" sx={{ border: 1, borderColor: 'divider' }}>
          <LogOutIcon fontSize="small" />
        </IconButton>
      </Box>
      
      <Divider sx={{ width: '100%', my: 3 }} />
      
      <Box sx={{ width: '100%' }}>
        <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Access Level</Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.light' }}>
            <ShieldIcon fontSize="small" />
          </Avatar>
          <Box>
            <Typography variant="body2" fontWeight="medium">Level 2</Typography>
            <Typography variant="caption" color="text.secondary">Standard Technician</Typography>
          </Box>
        </Box>
      </Box>
      
      <Divider sx={{ width: '100%', my: 3 }} />
      
      <Box sx={{ width: '100%' }}>
        <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Work Area</Typography>
        <Chip label="Assembly Line B" variant="outlined" />
      </Box>
    </Paper>
  );
}