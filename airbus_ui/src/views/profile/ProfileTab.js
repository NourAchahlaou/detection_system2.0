import * as React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  Divider 
} from '@mui/material';
import { 
  Edit as EditIcon,
  CheckCircle as CheckCircleIcon,
  Warning as AlertTriangleIcon,
  PhotoCamera as CameraIcon,
  TrendingUp as ActivityIcon
} from '@mui/icons-material';

export default function ProfileTab() {
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Personal Information</Typography>
        <Button 
          variant="text" 
          color="primary" 
          startIcon={<EditIcon />}
        >
          Edit Profile
        </Button>
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">First Name</Typography>
          <Typography variant="body1" fontWeight="medium">Achahlaou</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Last Name</Typography>
          <Typography variant="body1" fontWeight="medium">Nour</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Email</Typography>
          <Typography variant="body1" fontWeight="medium">nour.achahlaou@airbus.com</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Role</Typography>
          <Typography variant="body1" fontWeight="medium">Technician</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Shift Start</Typography>
          <Typography variant="body1" fontWeight="medium">08:00</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Shift End</Typography>
          <Typography variant="body1" fontWeight="medium">16:00</Typography>
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
            <Typography variant="body1" fontWeight="medium" color="success.main">Online</Typography>
          </Box>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Total Hours This Week</Typography>
          <Typography variant="body1" fontWeight="medium">31h 15min</Typography>
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="h6" sx={{ mb: 3 }}>Access & Credentials</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Employee ID</Typography>
          <Typography variant="body1" fontWeight="medium">T-49276</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Work Area</Typography>
          <Typography variant="body1" fontWeight="medium">Assembly Line B</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Badge Number</Typography>
          <Typography variant="body1" fontWeight="medium">AIR-8541</Typography>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Typography variant="caption" color="text.secondary">Access Level</Typography>
          <Typography variant="body1" fontWeight="medium">Level 2 (Standard Technician)</Typography>
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="h6" sx={{ mb: 3 }}>Summary Statistics</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: 'primary.light' }}>
            <CardContent>
              <CameraIcon color="primary" />
              <Typography variant="h4" fontWeight="bold" sx={{ my: 1 }}>278</Typography>
              <Typography variant="body2" color="text.secondary">Pieces Inspected Today</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: 'success.light' }}>
            <CardContent>
              <CheckCircleIcon color="success" />
              <Typography variant="h4" fontWeight="bold" sx={{ my: 1 }}>267</Typography>
              <Typography variant="body2" color="text.secondary">Verified Correct</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: 'error.light' }}>
            <CardContent>
              <AlertTriangleIcon color="error" />
              <Typography variant="h4" fontWeight="bold" sx={{ my: 1 }}>11</Typography>
              <Typography variant="body2" color="text.secondary">Issues Detected</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: 'secondary.light' }}>
            <CardContent>
              <ActivityIcon color="secondary" />
              <Typography variant="h4" fontWeight="bold" sx={{ my: 1 }}>96%</Typography>
              <Typography variant="body2" color="text.secondary">Detection Accuracy</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}