import * as React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  Typography,
  Box,
  Tabs,
  Tab,
  Card,
  CardContent,
  IconButton,
  Alert,
  CircularProgress,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  styled
} from '@mui/material';
import {
  Close as CloseIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Person as PersonIcon,
  Schedule as ScheduleIcon,
  DeleteForever as DeleteForeverIcon
} from '@mui/icons-material';
import { ProfileService } from './services/AccountService';
import { ShiftService } from './services/ShiftService';

// Initialize services
const profileService = new ProfileService();
const shiftService = new ShiftService();

// Styled Components
const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialog-paper': {
    borderRadius: '16px',
    borderColor: 'rgba(28, 53, 165, 0.26)',
  }
}));

const StyledDialogTitle = styled(DialogTitle)(({ theme }) => ({
  padding: '24px 24px 0 24px',
  borderBottom: 'none',
  '& .MuiTypography-root': {
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#2c3e50',
  }
}));

const StyledTabs = styled(Tabs)(({ theme }) => ({
  marginTop: '16px',
  borderBottom: '1px solid rgba(102, 126, 234, 0.1)',
  '& .MuiTabs-indicator': {
    backgroundColor: '#667eea',
    height: '3px',
    borderRadius: '3px',
  },
  '& .MuiTab-root': {
    textTransform: 'none',
    fontSize: '1rem',
    fontWeight: '600',
    color: '#64748b',
    minHeight: '48px',
    '&.Mui-selected': {
      color: '#667eea',
    },
    '&:hover': {
      color: '#667eea',
      backgroundColor: 'rgba(102, 126, 234, 0.04)',
    }
  },
}));

const StyledCard = styled(Card)(({ theme }) => ({
  borderRadius: '12px',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  backgroundColor: 'rgba(102, 126, 234, 0.1)',
  marginBottom: '24px',
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: '8px',
    '& fieldset': {
      borderColor: 'rgba(102, 126, 234, 0.2)',
    },
    '&:hover fieldset': {
      borderColor: 'rgba(102, 126, 234, 0.4)',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#667eea',
    },
  },
  '& .MuiInputLabel-root': {
    color: '#64748b',
    '&.Mui-focused': {
      color: '#667eea',
    },
  },
}));

const StyledFormControl = styled(FormControl)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: '8px',
    '& fieldset': {
      borderColor: 'rgba(102, 126, 234, 0.2)',
    },
    '&:hover fieldset': {
      borderColor: 'rgba(102, 126, 234, 0.4)',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#667eea',
    },
  },
  '& .MuiInputLabel-root': {
    color: '#64748b',
    '&.Mui-focused': {
      color: '#667eea',
    },
  },
}));

const ActionButton = styled(Button)(({ variant, color }) => ({
  textTransform: 'none',
  fontWeight: '600',
  borderRadius: '8px',
  padding: '8px 16px',
  backgroundColor: variant === 'contained' 
    ? color === 'error' ? '#ef4444' : '#667eea'
    : 'transparent',
  color: variant === 'contained' 
    ? 'white' 
    : color === 'error' ? '#ef4444' : '#667eea',
  border: variant === 'outlined' 
    ? `1px solid ${color === 'error' ? '#ef4444' : 'rgba(102, 126, 234, 0.3)'}` 
    : 'none',
  '&:hover': {
    backgroundColor: variant === 'contained' 
      ? color === 'error' ? '#dc2626' : '#5a67d8'
      : color === 'error' 
        ? 'rgba(239, 68, 68, 0.08)'
        : 'rgba(102, 126, 234, 0.08)',
  },
}));

const StyledTableContainer = styled(TableContainer)(({ theme }) => ({
  borderRadius: '12px',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  boxShadow: 'none',
  '& .MuiTableHead-root': {
    backgroundColor: 'rgba(102, 126, 234, 0.04)',
  },
  '& .MuiTableCell-head': {
    fontWeight: '700',
    color: '#475569',
    fontSize: '0.875rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    borderBottom: '2px solid rgba(102, 126, 234, 0.1)',
  },
  '& .MuiTableCell-body': {
    color: '#64748b',
    borderBottom: '1px solid rgba(102, 126, 234, 0.08)',
  },
  '& .MuiTableRow-root:hover': {
    backgroundColor: 'rgba(102, 126, 234, 0.02)',
  },
}));

const StatusChip = styled(Chip)(({ theme }) => ({
  fontSize: '0.75rem',
  fontWeight: '600',
  height: '28px',
  borderRadius: '8px',
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontSize: '1.25rem',
  fontWeight: '700',
  color: '#374151',
  marginBottom: '16px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
}));

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`profile-tabpanel-${index}`}
      aria-labelledby={`profile-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

export default function ProfileUpdateDialog({ 
  open, 
  onClose, 
  onProfileUpdate, 
  profileData 
}) {
  const [currentTab, setCurrentTab] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [shifts, setShifts] = React.useState([]);
  const [shiftsLoading, setShiftsLoading] = React.useState(false);
  
  // Profile form data
  const [profileForm, setProfileForm] = React.useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    airbusId: '',
    role: ''
  });
  
  // Shift form data
  const [shiftForm, setShiftForm] = React.useState({
    dayOfWeek: '',
    startTime: '',
    endTime: ''
  });
  
  const [editingShift, setEditingShift] = React.useState(null);
  const [notification, setNotification] = React.useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Initialize form data when dialog opens
  React.useEffect(() => {
    if (open && profileData) {
      setProfileForm({
        name: profileData.name || '',
        email: profileData.email || '',
        password: '',
        confirmPassword: '',
        airbusId: profileData.airbus_id || '',
        role: profileData.role || ''
      });
      loadShifts();
    }
  }, [open, profileData]);

  // Load user shifts
  const loadShifts = async () => {
    try {
      setShiftsLoading(true);
      const result = await shiftService.getAllUserShifts();
      if (result.success) {
        setShifts(result.shifts);
      } else {
        showNotification('Failed to load shifts', 'warning');
      }
    } catch (error) {
      console.error('Error loading shifts:', error);
      showNotification('Error loading shifts: ' + error.message, 'error');
    } finally {
      setShiftsLoading(false);
    }
  };

  // Show notification helper
  const showNotification = (message, severity) => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  // Handle profile form changes
  const handleProfileFormChange = (field) => (event) => {
    setProfileForm(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  // Handle shift form changes
  const handleShiftFormChange = (field) => (event) => {
    setShiftForm(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  // Validate profile form
  const validateProfileForm = () => {
    const validation = profileService.validateProfileData({
      name: profileForm.name,
      email: profileForm.email,
      password: profileForm.password || undefined,
      airbusId: profileForm.airbusId ? parseInt(profileForm.airbusId) : undefined,
      role: profileForm.role
    });

    if (!validation.isValid) {
      showNotification(validation.errors.join(', '), 'error');
      return false;
    }

    if (profileForm.password && profileForm.password !== profileForm.confirmPassword) {
      showNotification('Passwords do not match', 'error');
      return false;
    }

    return true;
  };

  // Validate shift form
  const validateShiftForm = () => {
    const validation = shiftService.validateShiftData({
      dayOfWeek: shiftForm.dayOfWeek,
      startTime: shiftForm.startTime,
      endTime: shiftForm.endTime
    });

    if (!validation.isValid) {
      showNotification(validation.errors.join(', '), 'error');
      return false;
    }

    return true;
  };

  // Handle profile update
  const handleUpdateProfile = async () => {
    if (!validateProfileForm()) return;

    try {
      setLoading(true);
      
      const updateData = {
        name: profileForm.name,
        email: profileForm.email,
        airbusId: profileForm.airbusId ? parseInt(profileForm.airbusId) : undefined,
        role: profileForm.role
      };

      // Only include password if it was changed
      if (profileForm.password) {
        updateData.password = profileForm.password;
      }

      const result = await profileService.updateUserProfile(updateData);
      
      if (result.success) {
        showNotification('Profile updated successfully', 'success');
        onProfileUpdate(result.profile);
      }
    } catch (error) {
      console.error('Error updating profile:', error);
      showNotification('Error updating profile: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Handle delete account
  const handleDeleteAccount = async () => {
    if (!window.confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      return;
    }

    try {
      setLoading(true);
      const result = await profileService.deleteUserAccount();
      
      if (result.success) {
        showNotification('Account deleted successfully', 'info');
        onProfileUpdate(null); // Signal account deletion
      }
    } catch (error) {
      console.error('Error deleting account:', error);
      showNotification('Error deleting account: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Handle add/update shift
  const handleSaveShift = async () => {
    if (!validateShiftForm()) return;

    try {
      setLoading(true);
      
      let result;
      if (editingShift) {
        result = await shiftService.updateShift(editingShift.id, {
          dayOfWeek: shiftForm.dayOfWeek,
          startTime: shiftForm.startTime,
          endTime: shiftForm.endTime
        });
      } else {
        result = await shiftService.createShift({
          dayOfWeek: shiftForm.dayOfWeek,
          startTime: shiftForm.startTime,
          endTime: shiftForm.endTime
        });
      }
      
      if (result.success) {
        showNotification(
          editingShift ? 'Shift updated successfully' : 'Shift created successfully', 
          'success'
        );
        setShiftForm({ dayOfWeek: '', startTime: '', endTime: '' });
        setEditingShift(null);
        loadShifts();
      }
    } catch (error) {
      console.error('Error saving shift:', error);
      showNotification('Error saving shift: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Handle edit shift
  const handleEditShift = (shift) => {
    setEditingShift(shift);
    setShiftForm({
      dayOfWeek: shift.day_of_week,
      startTime: shift.start_time,
      endTime: shift.end_time
    });
  };

  // Handle delete shift
  const handleDeleteShift = async (shiftId) => {
    if (!window.confirm('Are you sure you want to delete this shift?')) {
      return;
    }

    try {
      setLoading(true);
      const result = await shiftService.deleteShift(shiftId);
      
      if (result.success) {
        showNotification('Shift deleted successfully', 'success');
        loadShifts();
      }
    } catch (error) {
      console.error('Error deleting shift:', error);
      showNotification('Error deleting shift: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Handle cancel editing
  const handleCancelEdit = () => {
    setEditingShift(null);
    setShiftForm({ dayOfWeek: '', startTime: '', endTime: '' });
  };

  const dayOptions = [
    'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 
    'FRIDAY', 'SATURDAY', 'SUNDAY'
  ];

  const roleOptions = ['TECHN', 'ADMIN'];

  return (
    <>
      <StyledDialog 
        open={open} 
        onClose={onClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { height: '85vh' }
        }}
      >
        <StyledDialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>

            <Box sx={{ px: 3 }}>
              <StyledTabs value={currentTab} onChange={handleTabChange}>
                <Tab icon={<PersonIcon />} label="Profile Information" />
                <Tab icon={<ScheduleIcon />} label={`Work Schedule (${shifts.length})`} />
              </StyledTabs>
            </Box>
            <IconButton 
              onClick={onClose}
              sx={{ 
                color: '#64748b',
                '&:hover': { 
                  backgroundColor: 'rgba(102, 126, 234, 0.08)',
                  color: '#667eea'
                }
              }}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </StyledDialogTitle>


        <DialogContent sx={{ flex: 1, overflowY: 'auto',
            scrollbarWidth: 'none',
            '&::-webkit-scrollbar': { display: 'none' },
            msOverflowStyle: 'none', px: 3}}>
          <TabPanel value={currentTab} index={0}>
            <StyledCard>
              <CardContent sx={{ p: 3 }}>
                <SectionTitle>
                  <PersonIcon sx={{ color: '#667eea' }} />
                  Personal Information
                </SectionTitle>
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <StyledTextField
                      fullWidth
                      label="Full Name"
                      value={profileForm.name}
                      onChange={handleProfileFormChange('name')}
                      required
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <StyledTextField
                      fullWidth
                      label="Email Address"
                      type="email"
                      value={profileForm.email}
                      onChange={handleProfileFormChange('email')}
                      required
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <StyledTextField
                      fullWidth
                      label="New Password"
                      type="password"
                      value={profileForm.password}
                      onChange={handleProfileFormChange('password')}
                      helperText="Leave blank to keep current password"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <StyledTextField
                      fullWidth
                      label="Confirm New Password"
                      type="password"
                      value={profileForm.confirmPassword}
                      onChange={handleProfileFormChange('confirmPassword')}
                      disabled={!profileForm.password}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <StyledTextField
                      fullWidth
                      label="Airbus ID"
                      type="number"
                      value={profileForm.airbusId}
                      onChange={handleProfileFormChange('airbusId')}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <StyledFormControl fullWidth>
                      <InputLabel>Role</InputLabel>
                      <Select
                        value={profileForm.role}
                        onChange={handleProfileFormChange('role')}
                        label="Role"
                      >
                        <MenuItem value="AUDITOR">Auditor</MenuItem>
                        <MenuItem value="TECHNICIAN">Technician</MenuItem>
                      </Select>
                    </StyledFormControl>
                  </Grid>
                </Grid>
              </CardContent>
            </StyledCard>

            <StyledCard>
              <CardContent sx={{ p: 3 }}>
                <SectionTitle sx={{ color: '#ef4444' }}>
                  <DeleteForeverIcon sx={{ color: '#ef4444' }} />
                  Danger Zone
                </SectionTitle>
                
                <Alert severity="warning" sx={{ mb: 3, borderRadius: '8px' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: '600', mb: 1 }}>
                    Account Deletion
                  </Typography>
                  <Typography variant="body2">
                    Deleting your account will permanently remove all your data including shifts and cannot be undone.
                  </Typography>
                </Alert>
                
                <ActionButton
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteForeverIcon />}
                  onClick={handleDeleteAccount}
                  disabled={loading}
                >
                  Delete Account
                </ActionButton>
              </CardContent>
            </StyledCard>
          </TabPanel>

          <TabPanel value={currentTab} index={1}>
            <StyledCard>
              <CardContent sx={{ p: 3 }}>
                <SectionTitle>
                  <AddIcon sx={{ color: '#667eea' }} />
                  {editingShift ? 'Edit Shift' : 'Add New Shift'}
                </SectionTitle>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={4}>
                    <StyledFormControl fullWidth>
                      <InputLabel>Day of Week</InputLabel>
                      <Select
                        value={shiftForm.dayOfWeek}
                        onChange={handleShiftFormChange('dayOfWeek')}
                        label="Day of Week"
                      >
                        {dayOptions.map(day => (
                          <MenuItem key={day} value={day}>
                            {day.charAt(0) + day.slice(1).toLowerCase()}
                          </MenuItem>
                        ))}
                      </Select>
                    </StyledFormControl>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <StyledTextField
                      fullWidth
                      label="Start Time"
                      type="time"
                      value={shiftForm.startTime}
                      onChange={handleShiftFormChange('startTime')}
                      InputLabelProps={{ shrink: true }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <StyledTextField
                      fullWidth
                      label="End Time"
                      type="time"
                      value={shiftForm.endTime}
                      onChange={handleShiftFormChange('endTime')}
                      InputLabelProps={{ shrink: true }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Stack direction="row" spacing={2}>
                      <ActionButton
                        variant="contained"
                        startIcon={editingShift ? <SaveIcon /> : <AddIcon />}
                        onClick={handleSaveShift}
                        disabled={loading || !shiftForm.dayOfWeek || !shiftForm.startTime || !shiftForm.endTime}
                      >
                        {editingShift ? 'Update Shift' : 'Add Shift'}
                      </ActionButton>
                      {editingShift && (
                        <ActionButton
                          variant="outlined"
                          startIcon={<CancelIcon />}
                          onClick={handleCancelEdit}
                        >
                          Cancel
                        </ActionButton>
                      )}
                    </Stack>
                  </Grid>
                </Grid>
              </CardContent>
            </StyledCard>

            <Box sx={{ mb: 2 }}>
              <SectionTitle>
                <ScheduleIcon sx={{ color: '#667eea' }} />
                Current Schedule ({shifts.length} shifts)
              </SectionTitle>
            </Box>
            
            {shiftsLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
                <CircularProgress sx={{ color: '#667eea' }} />
              </Box>
            ) : shifts.length === 0 ? (
              <Alert 
                severity="info" 
                sx={{ borderRadius: '12px', py: 3 }}
              >
                <Typography variant="subtitle2" sx={{ fontWeight: '600', mb: 1 }}>
                  No shifts configured
                </Typography>
                <Typography variant="body2">
                  Add your first shift using the form above to get started.
                </Typography>
              </Alert>
            ) : (
              <StyledTableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Day</TableCell>
                      <TableCell>Start Time</TableCell>
                      <TableCell>End Time</TableCell>
                      <TableCell align="center">Duration</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {shifts.map((shift) => {
                      // Calculate duration
                      const startTime = new Date(`2000-01-01T${shift.start_time}`);
                      const endTime = new Date(`2000-01-01T${shift.end_time}`);
                      const durationMs = endTime.getTime() - startTime.getTime();
                      const durationHours = Math.floor(durationMs / (1000 * 60 * 60));
                      const durationMinutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
                      const durationText = `${durationHours}h ${durationMinutes > 0 ? durationMinutes + 'm' : ''}`;

                      return (
                        <TableRow key={shift.id}>
                          <TableCell>
                            <StatusChip 
                              label={shift.day_name} 
                              sx={{ 
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                color: '#667eea',
                                fontWeight: '600'
                              }}
                            />
                          </TableCell>
                          <TableCell sx={{ fontWeight: '500' }}>
                            {shift.start_time}
                          </TableCell>
                          <TableCell sx={{ fontWeight: '500' }}>
                            {shift.end_time}
                          </TableCell>
                          <TableCell align="center">
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                color: '#64748b',
                                fontWeight: '500'
                              }}
                            >
                              {durationText}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Stack direction="row" spacing={1} justifyContent="center">
                              <IconButton
                                size="small"
                                onClick={() => handleEditShift(shift)}
                                disabled={loading}
                                sx={{
                                  color: '#667eea',
                                  '&:hover': {
                                    backgroundColor: 'rgba(102, 126, 234, 0.08)'
                                  }
                                }}
                              >
                                <EditIcon fontSize="small" />
                              </IconButton>
                              <IconButton
                                size="small"
                                onClick={() => handleDeleteShift(shift.id)}
                                disabled={loading}
                                sx={{
                                  color: '#ef4444',
                                  '&:hover': {
                                    backgroundColor: 'rgba(239, 68, 68, 0.08)'
                                  }
                                }}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Stack>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </StyledTableContainer>
            )}
          </TabPanel>
        </DialogContent>

        <DialogActions sx={{ p: 3, borderTop: '1px solid rgba(102, 126, 234, 0.1)' }}>
          <ActionButton onClick={onClose} disabled={loading} variant="outlined">
            Close
          </ActionButton>
          {currentTab === 0 && (
            <ActionButton 
              variant="contained" 
              onClick={handleUpdateProfile}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : <SaveIcon />}
            >
              {loading ? 'Updating...' : 'Update Profile'}
            </ActionButton>
          )}
        </DialogActions>
      </StyledDialog>

      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={() => setNotification({ ...notification, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setNotification({ ...notification, open: false })} 
          severity={notification.severity} 
          sx={{ 
            width: '100%',
            borderRadius: '8px'
          }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
}