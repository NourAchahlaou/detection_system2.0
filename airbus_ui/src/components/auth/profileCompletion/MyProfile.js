import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom'; // Import the useNavigate hook
import api from '../../../utils/UseAxios';
import {
  Box,
  Button,
  Divider,
  FormControl,
  FormLabel,
  Select,
  Typography,
  Grid,
  TextField,
  IconButton,
  MenuItem,
  Card,
  CardContent,
  Chip,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Snackbar,
  Alert,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  AddCircleOutline, 
  DeleteOutline, 
  AccessTimeOutlined,
  CheckCircle,
  PendingOutlined,
} from '@mui/icons-material';

// Custom styled components (kept from original)
const ProfileCard = styled(Card)(({ theme }) => ({
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'center',
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  boxShadow:
    'hsla(225, 30.80%, 5.10%, 0.05) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.05) 0px 15px 35px -5px',
  [theme.breakpoints.up('sm')]: {
    maxWidth: '800px',
  },
  margin: 'auto',
  ...theme.applyStyles('dark', {
    backgroundColor: 'hsla(220, 35%, 3%, 0.4)',
    boxShadow:
      'hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px',
  }),
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(2),
}));

// Custom circular progress with percentage
const CircularProgressWithLabel = ({ value, size = 120 }) => {
  return (
    <Box sx={{ position: 'relative', display: 'inline-flex', justifyContent: 'center', width: '100%' }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          variant="determinate"
          value={100}
          size={size}
          thickness={4}
          sx={{ color: (theme) => theme.palette.grey[200] }}
        />
        <CircularProgress
          variant="determinate"
          value={value}
          size={size}
          thickness={4}
          sx={{
            color: (theme) => value < 30 
              ? theme.palette.error.main 
              : value < 70 
                ? theme.palette.warning.main 
                : theme.palette.success.main,
            position: 'absolute',
            left: 0,
          }}
        />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography variant="h4" component="div" color="text.primary" fontWeight="bold">
            {`${Math.round(value)}%`}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

// Map day values to ShiftDay enum values in backend
const dayValueToEnum = {
  0: 'MONDAY',
  1: 'TUESDAY',
  2: 'WEDNESDAY',
  3: 'THURSDAY',
  4: 'FRIDAY',
  5: 'SATURDAY',
  6: 'SUNDAY',
};

// Mapping for display purposes
const days = [
  { label: 'Monday', value: 0 },
  { label: 'Tuesday', value: 1 },
  { label: 'Wednesday', value: 2 },
  { label: 'Thursday', value: 3 },
  { label: 'Friday', value: 4 },
  { label: 'Saturday', value: 5 },
  { label: 'Sunday', value: 6 },
];

function MyProfile() {
  // Initialize navigate function from React Router
  const navigate = useNavigate();
  
  // Form state
  const [airbusId, setAirbusId] = useState('');
  const [role, setRole] = useState('');
  const [mainShiftDays, setMainShiftDays] = useState([]);
  const [mainShift, setMainShift] = useState({ start: '', end: '' });
  const [specialShifts, setSpecialShifts] = useState([]);
  const [savedSuccessfully, setSavedSuccessfully] = useState(false);
  // API related state
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [initialCompletionData, setInitialCompletionData] = useState({
    completion_percentage: 0,
    completion_status: {
      personal_info: {
        airbus_id: false,
        role: false
      },
      main_shift: {
        shifts: false,
        has_weekday_coverage: false
      }
    },
    missing_fields: []
  });
  
  // Live completion status (updated in real-time)
  const [liveCompletionData, setLiveCompletionData] = useState({
    completion_percentage: 0,
    completion_status: {
      personal_info: {
        airbus_id: false,
        role: false
      },
      main_shift: {
        shifts: false,
        has_weekday_coverage: false
      }
    },
    missing_fields: []
  });
  
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Function to show notifications
  const showNotification = useCallback((message, severity) => {
    setNotification({
      open: true,
      message,
      severity
    });
  }, []);
  
  // Function to update live completion status based on current form state
  const updateLiveCompletionStatus = useCallback(() => {
    // Create a copy of the current completion data
    const updatedCompletionData = { 
      ...liveCompletionData,
      completion_status: {
        personal_info: {
          airbus_id: Boolean(airbusId.trim()),
          role: Boolean(role)
        },
        main_shift: {
          shifts: mainShiftDays.length > 0 && mainShift.start && mainShift.end,
          // Check if there's at least one weekday (Monday-Friday) in mainShiftDays
          has_weekday_coverage: mainShiftDays.some(day => day >= 0 && day <= 4)
        }
      }
    };
    
    // Calculate missing fields
    const missingFields = [];
    
    if (!updatedCompletionData.completion_status.personal_info.airbus_id) {
      missingFields.push('Airbus ID');
    }
    
    if (!updatedCompletionData.completion_status.personal_info.role) {
      missingFields.push('Role');
    }
    
    if (!updatedCompletionData.completion_status.main_shift.shifts) {
      missingFields.push('Work Schedule');
    }
    
    if (!updatedCompletionData.completion_status.main_shift.has_weekday_coverage) {
      missingFields.push('Weekday Work Schedule');
    }
    
    updatedCompletionData.missing_fields = missingFields;
    
    // Calculate completion percentage (4 items to complete)
    const completedItems = Object.values(updatedCompletionData.completion_status.personal_info).filter(Boolean).length + 
                           Object.values(updatedCompletionData.completion_status.main_shift).filter(Boolean).length;
    
    updatedCompletionData.completion_percentage = (completedItems / 4) * 100;
    
    setLiveCompletionData(updatedCompletionData);
    
  }, [airbusId, role, mainShiftDays, mainShift, liveCompletionData]);
  
  // Effect to update live completion status whenever form fields change
  useEffect(() => {
    updateLiveCompletionStatus();
  }, [airbusId, role, mainShiftDays, mainShift, updateLiveCompletionStatus]);
  
  const fetchUserProfile = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/users/profile/profile');
      const profileData = response.data;
      
      // Update form state with fetched data
      setAirbusId(profileData.airbus_id || '');
      setRole(profileData.role || '');
      
      // Process shifts data
      if (profileData.shifts && profileData.shifts.length > 0) {
        // Group shifts by day (assuming backend provides day as enum string)
        const dayMap = {
          'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 
          'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        };
        
        // Extract unique days for main shift
        const uniqueDays = [...new Set(profileData.shifts.map(shift => dayMap[shift.day]))];
        setMainShiftDays(uniqueDays);
        
        // Set main shift times using the first shift as reference
        if (profileData.shifts[0]) {
          setMainShift({
            start: profileData.shifts[0].start_time,
            end: profileData.shifts[0].end_time
          });
        }
      }
      
      // Store initial completion data from the backend
      if (profileData.completion) {
        setInitialCompletionData(profileData.completion);
        setLiveCompletionData(profileData.completion); // Initialize live data with backend data
      }
      
    } catch (error) {
      console.error('Error fetching profile:', error);
      showNotification('Failed to load profile data', 'error');
    } finally {
      setLoading(false);
    }
  }, [showNotification]); 

  useEffect(() => {
    fetchUserProfile();
  }, [fetchUserProfile]);

// Updated handleSaveProfile function with navigation to dashboard after successful update
 useEffect(() => {
    if (savedSuccessfully) {
      // Navigate to dashboard
      navigate('/dashboard', { replace: true });
    }
  }, [savedSuccessfully, navigate]);
  
  const handleSaveProfile = async () => {
    setSaving(true);
    
    try {
      // Form validation
      if (!airbusId) {
        showNotification('Please enter your Airbus ID', 'error');
        return;
      }
      
      if (!role) {
        showNotification('Please select your role', 'error');
        return;
      }
      
      if (mainShiftDays.length === 0 || !mainShift.start || !mainShift.end) {
        showNotification('Please set your main shift schedule', 'error');
        return;
      }
      
      // Create shifts from main shift days
      const shifts = mainShiftDays.map(dayValue => ({
        day_of_week: dayValueToEnum[dayValue],
        start_time: mainShift.start,
        end_time: mainShift.end
      }));
      
      // Add special shifts if any
      specialShifts.forEach(shift => {
        if (shift.day !== '' && shift.start && shift.end) {
          shifts.push({
            day_of_week: dayValueToEnum[shift.day],
            start_time: shift.start,
            end_time: shift.end
          });
        }
      });
      
      // Prepare request data with proper type conversions
      const profileData = {
        airbus_id: parseInt(airbusId, 10),
        role: role,
        main_shifts: shifts
      };
      
      console.log("Sending profile data:", profileData);
      
      // Send update request
      await api.put('/api/users/profile/update', profileData);
      
      // Update the initialCompletionData with our current live completion data
      setInitialCompletionData(liveCompletionData);
      
      showNotification('Profile updated successfully', 'success');
      
      // Set saved flag to trigger navigation effect
      setSavedSuccessfully(true);
      
    } catch (error) {
      console.error('Error updating profile:', error);
      
      let errorMessage = 'Failed to update profile';
      
      // Safely extract error message
      if (error.response?.data?.detail) {
        errorMessage = typeof error.response.data.detail === 'string' ?
          error.response.data.detail :
          'Failed to update profile';
      }
      
      showNotification(errorMessage, 'error');
    } finally {
      setSaving(false);
    }
  };
  

  const handleAddSpecialShift = () => {
    setSpecialShifts([
      ...specialShifts,
      { day: '', start: '', end: '' }
    ]);
  };

  const handleSpecialShiftChange = (index, field, value) => {
    const updated = [...specialShifts];
    updated[index][field] = value;
    setSpecialShifts(updated);
  };

  const handleRemoveSpecialShift = (index) => {
    const updated = [...specialShifts];
    updated.splice(index, 1);
    setSpecialShifts(updated);
  };

  const renderDayChips = () => {
    return mainShiftDays.map(dayValue => {
      const dayObj = days.find(d => d.value === dayValue);
      return dayObj ? (
        <Chip 
          key={dayObj.value} 
          label={dayObj.label} 
          color="primary" 
          variant="outlined" 
          size="small" 
          sx={{ m: 0.5 }}
        />
      ) : null;
    });
  };
  
  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false 
    });
  };

  return (
    <Box sx={{ 
      flex: 1, 
      display: 'flex', 
      flexDirection: 'column',
      py: 4,
      px: 2,
    }}>
      {/* Progress Stepper */}
      <Box sx={{ width: '100%', maxWidth: '1000px', mx: 'auto', mb: 4 }}>
        <Stepper activeStep={1} alternativeLabel>
          <Step completed={true}>
            <StepLabel>Sign Up</StepLabel>
          </Step>
          <Step active={true}>
            <StepLabel>Complete Profile</StepLabel>
          </Step>
          <Step>
            <StepLabel>Start Using Platform</StepLabel>
          </Step>
        </Stepper>
      </Box>
      
      {/* Main Content */}
      <Box sx={{ 
        display: 'flex', 
        width: '100%', 
        maxWidth: '1000px', 
        mx: 'auto',
        flexDirection: { xs: 'column', md: 'row' },
        gap: 3
      }}>
        {/* Main Profile Form Card */}
        <ProfileCard sx={{ flex: 3 }}>
          <CardContent sx={{ p: 2 }}>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
              Complete Your Profile
            </Typography>
            <Divider sx={{ mb: 4 }} />

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {/* Personal Information Section */}
                <Box component="form" sx={{ mb: 4 }}>
                  <SectionTitle variant="h6">
                    Personal Information
                  </SectionTitle>
                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <FormLabel htmlFor="airbusId" sx={{ mb: 1 }}>Airbus ID</FormLabel>
                        <TextField
                          id="airbusId"
                          fullWidth
                          placeholder="Enter your Airbus ID"
                          value={airbusId}
                          onChange={(e) => setAirbusId(e.target.value)}
                          variant="outlined"
                          error={liveCompletionData.missing_fields.includes('Airbus ID')}
                          helperText={liveCompletionData.missing_fields.includes('Airbus ID') ? 'Required field' : ''}
                        />
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <FormLabel htmlFor="role" sx={{ mb: 1 }}>Role</FormLabel>
                        <Select
                          id="role"
                          value={role}
                          onChange={(e) => setRole(e.target.value)}
                          displayEmpty
                          placeholder="Select your role"
                          error={liveCompletionData.missing_fields.includes('Role')}
                        >
                          <MenuItem value="" disabled>Select your role</MenuItem>
                          <MenuItem value="OBSERVER">Observer</MenuItem>
                          <MenuItem value="TECHNICIAN">Technician</MenuItem>
                          <MenuItem value="ADMIN">Admin</MenuItem>
                        </Select>
                        {liveCompletionData.missing_fields.includes('Role') && (
                          <Typography variant="caption" color="error">Required field</Typography>
                        )}
                      </FormControl>
                    </Grid>
                  </Grid>
                </Box>

                <Divider sx={{ my: 4 }} />

                {/* Main Shift Section */}
                <Box sx={{ mb: 4 }}>
                  <SectionTitle variant="h6">
                    <AccessTimeOutlined color="primary" />
                    Main Shift Schedule
                  </SectionTitle>
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <FormControl fullWidth>
                        <FormLabel htmlFor="workingDays" sx={{ mb: 1 }}>Working Days</FormLabel>
                        <Select
                          id="workingDays"
                          multiple
                          value={mainShiftDays}
                          onChange={(e) => setMainShiftDays(e.target.value)}
                          renderValue={() => (
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                              {renderDayChips()}
                            </Box>
                          )}
                          MenuProps={{
                            PaperProps: {
                              style: {
                                maxHeight: 224,
                              },
                            },
                          }}
                          error={liveCompletionData.missing_fields.includes('Work Schedule') || 
                                liveCompletionData.missing_fields.includes('Weekday Work Schedule')}
                        >
                          {days.map((day) => (
                            <MenuItem key={day.value} value={day.value}>
                              {day.label}
                            </MenuItem>
                          ))}
                        </Select>
                        {liveCompletionData.missing_fields.includes('Weekday Work Schedule') && (
                          <Typography variant="caption" color="error">At least one weekday is required</Typography>
                        )}
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <FormLabel htmlFor="startTime" sx={{ mb: 1 }}>Start Time</FormLabel>
                        <TextField
                          id="startTime"
                          type="time"
                          value={mainShift.start}
                          onChange={(e) => setMainShift({ ...mainShift, start: e.target.value })}
                          fullWidth
                          InputLabelProps={{ shrink: true }}
                          error={liveCompletionData.missing_fields.includes('Work Schedule')}
                        />
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <FormLabel htmlFor="endTime" sx={{ mb: 1 }}>End Time</FormLabel>
                        <TextField
                          id="endTime"
                          type="time"
                          value={mainShift.end}
                          onChange={(e) => setMainShift({ ...mainShift, end: e.target.value })}
                          fullWidth
                          InputLabelProps={{ shrink: true }}
                          error={liveCompletionData.missing_fields.includes('Work Schedule')}
                        />
                      </FormControl>
                    </Grid>
                  </Grid>
                </Box>

                {/* Special Shifts Section */}
                <Box sx={{ mb: 4 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <SectionTitle variant="h6" sx={{ mb: 0 }}>
                      <AccessTimeOutlined color="secondary" />
                      Special Shifts
                    </SectionTitle>
                    <Button 
                      variant="outlined" 
                      startIcon={<AddCircleOutline />}
                      onClick={handleAddSpecialShift}
                      color="secondary"
                      size="small"
                    >
                      Add Shift
                    </Button>
                  </Box>

                  {specialShifts.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', mt: 2 }}>
                      No special shifts added. Click the button above to add one.
                    </Typography>
                  ) : (
                    specialShifts.map((shift, index) => (
                      <Card 
                        key={index} 
                        variant="outlined" 
                        sx={{ 
                          mb: 2, 
                          p: 2,
                          borderColor: 'divider',
                          '&:hover': {
                            borderColor: 'secondary.main',
                          }
                        }}
                      >
                        <Grid container spacing={2} alignItems="center">
                          <Grid item xs={12} sm={4}>
                            <FormControl fullWidth>
                              <FormLabel sx={{ mb: 1 }}>Day</FormLabel>
                              <Select
                                value={shift.day}
                                onChange={(e) => handleSpecialShiftChange(index, 'day', e.target.value)}
                                displayEmpty
                              >
                                <MenuItem value="" disabled>Select a day</MenuItem>
                                {days.map((day) => (
                                  <MenuItem key={day.value} value={day.value}>{day.label}</MenuItem>
                                ))}
                              </Select>
                            </FormControl>
                          </Grid>
                          <Grid item xs={12} sm={3}>
                            <FormControl fullWidth>
                              <FormLabel sx={{ mb: 1 }}>Start</FormLabel>
                              <TextField
                                type="time"
                                value={shift.start}
                                onChange={(e) => handleSpecialShiftChange(index, 'start', e.target.value)}
                                fullWidth
                                InputLabelProps={{ shrink: true }}
                              />
                            </FormControl>
                          </Grid>
                          <Grid item xs={12} sm={3}>
                            <FormControl fullWidth>
                              <FormLabel sx={{ mb: 1 }}>End</FormLabel>
                              <TextField
                                type="time"
                                value={shift.end}
                                onChange={(e) => handleSpecialShiftChange(index, 'end', e.target.value)}
                                fullWidth
                                InputLabelProps={{ shrink: true }}
                              />
                            </FormControl>
                          </Grid>
                          <Grid item xs={12} sm={2} sx={{ display: 'flex', justifyContent: 'center' }}>
                            <IconButton 
                              onClick={() => handleRemoveSpecialShift(index)}
                              color="error"
                              size="medium"
                              sx={{ 
                                mt: { xs: 0, sm: 3 },
                                '&:hover': { backgroundColor: 'error.light', color: 'white' }
                              }}
                            >
                              <DeleteOutline />
                            </IconButton>
                          </Grid>
                        </Grid>
                      </Card>
                    ))
                  )}
                </Box>

                <Divider sx={{ my: 4 }} />
                
                {/* Footer Buttons */}
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 2 }}>
                  <Button 
                    variant="outlined" 
                    color="inherit"
                    size="large"
                    onClick={() => window.history.back()}
                  >
                    Cancel
                  </Button>
                  <Button 
                    variant="contained" 
                    color="primary"
                    size="large"
                    onClick={handleSaveProfile}
                    disabled={saving}
                  >
                    {saving ? 'Saving...' : 'Save Profile'}
                  </Button>
                </Box>
              </>
            )}
          </CardContent>
        </ProfileCard>
        
        {/* Completion Status Card */}
        <Box sx={{ 
          flex: 1, 
          height: 'fit-content',
          position: { md: 'sticky' },
          top: { md: '2rem' },
          p: 2
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <CircularProgressWithLabel 
                value={liveCompletionData.completion_percentage} 
              />
            </Box>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 500, textAlign: 'center' }}>
              Profile Completion
            </Typography>
            <Card sx={(theme) => ({
              p: 2,
              ...theme.applyStyles('dark', {
                backgroundColor: 'hsla(220, 35%, 3%, 0.4)',
                boxShadow:
                  'hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px',
              }),
            })}>
              <List sx={{ py: 0 }}>
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {liveCompletionData.completion_status?.personal_info?.airbus_id ? 
                      <CheckCircle color="success" fontSize="small" /> : 
                      <PendingOutlined color="action" fontSize="small" />
                    }
                  </ListItemIcon>
                  <ListItemText 
                    primary="Airbus ID" 
                    secondary={liveCompletionData.completion_status?.personal_info?.airbus_id ? "Completed" : "Pending"}
                    secondaryTypographyProps={{ 
                      color: liveCompletionData.completion_status?.personal_info?.airbus_id ? "success.main" : "text.secondary" 
                    }}
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {liveCompletionData.completion_status?.personal_info?.role ? 
                      <CheckCircle color="success" fontSize="small" /> : 
                      <PendingOutlined color="action" fontSize="small" />
                    }
                  </ListItemIcon>
                  <ListItemText 
                    primary="Role Selection" 
                    secondary={liveCompletionData.completion_status?.personal_info?.role ? "Completed" : "Pending"}
                    secondaryTypographyProps={{ 
                      color: liveCompletionData.completion_status?.personal_info?.role ? "success.main" : "text.secondary" 
                    }}
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {liveCompletionData.completion_status?.main_shift?.shifts ? 
                      <CheckCircle color="success" fontSize="small" /> : 
                      <PendingOutlined color="action" fontSize="small" />
                    }
                  </ListItemIcon>
                  <ListItemText 
                    primary="Working Days & Hours" 
                    secondary={liveCompletionData.completion_status?.main_shift?.shifts ? "Completed" : "Pending"}
                    secondaryTypographyProps={{ 
                      color: liveCompletionData.completion_status?.main_shift?.shifts ? "success.main" : "text.secondary" 
                    }}
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {liveCompletionData.completion_status?.main_shift?.has_weekday_coverage ? 
                      <CheckCircle color="success" fontSize="small" /> : 
                      <PendingOutlined color="action" fontSize="small" />
                    }
                  </ListItemIcon>
                  <ListItemText 
                    primary="Weekday Coverage" 
                    secondary={liveCompletionData.completion_status?.main_shift?.has_weekday_coverage ? "Completed" : "Pending"}
                    secondaryTypographyProps={{ 
                      color: liveCompletionData.completion_status?.main_shift?.has_weekday_coverage ? "success.main" : "text.secondary" 
                    }}
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Special Shifts" 
                    secondary="Optional"
                    secondaryTypographyProps={{ color: "success.main" }}
                  />
                </ListItem>
              </List>
              
              <Divider sx={{ my: 2 }} />
              
              {liveCompletionData.missing_fields && liveCompletionData.missing_fields.length > 0 && (
                <>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Missing Fields:</Typography>
                  <List dense>
                    {liveCompletionData.missing_fields.map((field, index) => (
                      <ListItem key={index} sx={{ py: 0.5 }}>
                        <Typography variant="body2" color="error">• {field}</Typography>
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
              
              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', textAlign: 'center', mt: 2 }}>
                Complete all required fields to unlock full platform access
              </Typography>
            </Card>
          </CardContent>
        </Box>
      </Box>
      
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
    </Box>
  );
}

export default MyProfile; 