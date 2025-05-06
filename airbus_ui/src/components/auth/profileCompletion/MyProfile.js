import React, { useState, useEffect } from 'react';
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
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  AddCircleOutline, 
  DeleteOutline, 
  AccessTimeOutlined,
  CheckCircle,
  PendingOutlined,
  
} from '@mui/icons-material';

// Custom styled components
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

  // 💡 Remove the static backgroundColor
  // ✅ And instead apply conditional dark mode background:
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
  const [airbusId, setAirbusId] = useState('');
  const [role, setRole] = useState('');
  const [mainShiftDays, setMainShiftDays] = useState([]);
  const [mainShift, setMainShift] = useState({ start: '', end: '' });
  const [specialShifts, setSpecialShifts] = useState([]);
  const [completionPercentage, setCompletionPercentage] = useState(0);
  const [completionStatus, setCompletionStatus] = useState({
    personalInfo: {
      airbusId: false,
      role: false
    },
    mainShift: {
      days: false,
      times: false
    },
    specialShifts: true // true by default as it's optional
  });

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
  
  // Calculate completion percentage
  useEffect(() => {
    // Update status for each field
    const newCompletionStatus = {
      personalInfo: {
        airbusId: !!airbusId.trim(),
        role: !!role
      },
      mainShift: {
        days: mainShiftDays.length > 0,
        times: !!mainShift.start && !!mainShift.end
      },
      specialShifts: true // Always true as it's optional
    };
    
    setCompletionStatus(newCompletionStatus);
    
    // Calculate percentage
    const totalFields = 4; // airbusId, role, mainShift days, mainShift times
    const completedFields = [
      newCompletionStatus.personalInfo.airbusId,
      newCompletionStatus.personalInfo.role,
      newCompletionStatus.mainShift.days,
      newCompletionStatus.mainShift.times
    ].filter(Boolean).length;
    
    setCompletionPercentage(Math.round((completedFields / totalFields) * 100));
  }, [airbusId, role, mainShiftDays, mainShift]);

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
                  >
                    <MenuItem value="" disabled>Select your role</MenuItem>
                    <MenuItem value="OBSERVER">Observer</MenuItem>
                    <MenuItem value="TECHNICIAN">Technician</MenuItem>
                    <MenuItem value="ADMIN">Admin</MenuItem>
                  </Select>
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
                  >
                    {days.map((day) => (
                      <MenuItem key={day.value} value={day.value}>
                        {day.label}
                      </MenuItem>
                    ))}
                  </Select>
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
            >
              Cancel
            </Button>
            <Button 
              variant="contained" 
              color="primary"
              size="large"
            >
              Save Profile
            </Button>
          </Box>
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
            <Box sx={{ width: '100%', mr: 1 }}>
              <CircularProgressWithLabel 
                variant="determinate" 
                value={completionPercentage} 
                sx={{ 
                  height: 8, 
                  borderRadius: 2,
                  backgroundColor: 'gray.200'
                }}
              />
            </Box>

          </Box>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 , textAlign :'center'}}>
            Profile Completion
          </Typography>
          <Card   sx={(theme) => ({
              flex: 1,
              height: 'fit-content',
              position: { md: 'sticky' },
              top: { md: '2rem' },
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
                {completionStatus.personalInfo.airbusId ? 
                  <CheckCircle color="success" fontSize="small" /> : 
                  <PendingOutlined color="action" fontSize="small" />
                }
              </ListItemIcon>
              <ListItemText 
                primary="Airbus ID" 
                secondary={completionStatus.personalInfo.airbusId ? "Completed" : "Pending"}
                secondaryTypographyProps={{ 
                  color: completionStatus.personalInfo.airbusId ? "success.main" : "text.secondary" 
                }}
              />
            </ListItem>
            
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {completionStatus.personalInfo.role ? 
                  <CheckCircle color="success" fontSize="small" /> : 
                  <PendingOutlined color="action" fontSize="small" />
                }
              </ListItemIcon>
              <ListItemText 
                primary="Role Selection" 
                secondary={completionStatus.personalInfo.role ? "Completed" : "Pending"}
                secondaryTypographyProps={{ 
                  color: completionStatus.personalInfo.role ? "success.main" : "text.secondary" 
                }}
              />
            </ListItem>
            
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {completionStatus.mainShift.days ? 
                  <CheckCircle color="success" fontSize="small" /> : 
                  <PendingOutlined color="action" fontSize="small" />
                }
              </ListItemIcon>
              <ListItemText 
                primary="Working Days" 
                secondary={completionStatus.mainShift.days ? "Completed" : "Pending"}
                secondaryTypographyProps={{ 
                  color: completionStatus.mainShift.days ? "success.main" : "text.secondary" 
                }}
              />
            </ListItem>
            
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {completionStatus.mainShift.times ? 
                  <CheckCircle color="success" fontSize="small" /> : 
                  <PendingOutlined color="action" fontSize="small" />
                }
              </ListItemIcon>
              <ListItemText 
                primary="Working Hours" 
                secondary={completionStatus.mainShift.times ? "Completed" : "Pending"}
                secondaryTypographyProps={{ 
                  color: completionStatus.mainShift.times ? "success.main" : "text.secondary" 
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
          
          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', textAlign: 'center' }}>
            Complete all required fields to unlock full platform access
          </Typography>
          </Card>
        </CardContent>
      </Box>
    </Box>
  </Box>
  );
}

export default MyProfile;