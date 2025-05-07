import * as React from 'react';
import { useState } from 'react';
import { 
  Grid, 
  Box, 
  Paper, 
  Typography, 
  Tabs, 
  Tab, 
  Avatar, 
  Chip, 
  Button, 
  Card, 
  CardContent, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  TablePagination,
  Stack,
  IconButton,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  Switch,
  TextField
} from '@mui/material';
import { 
  Schedule as ClockIcon, 
  Notifications as BellIcon, 
  TrendingUp as ActivityIcon, 
  Person as UserIcon, 
  Settings as SettingsIcon, 
  CheckCircle as CheckCircleIcon, 
  Warning as AlertTriangleIcon, 
  PhotoCamera as CameraIcon, 
  BarChart as BarChartIcon, 
  Description as FileTextIcon, 
  Visibility as EyeIcon, 
  Edit as EditIcon, 
  Logout as LogOutIcon, 
  Group as UsersIcon, 
  Security as ShieldIcon, 
  FindInPage as FileSearchIcon,
  CalendarToday as CalendarIcon,
  Email as EmailIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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
  
  // Dummy data for the Performance tab charts
  const performanceData = [
    { day: 'Mon', inspected: 45, issues: 3 },
    { day: 'Tue', inspected: 52, issues: 5 },
    { day: 'Wed', inspected: 49, issues: 2 },
    { day: 'Thu', inspected: 50, issues: 4 },
    { day: 'Fri', inspected: 43, issues: 3 },
    { day: 'Sat', inspected: 30, issues: 1 },
    { day: 'Today', inspected: 25, issues: 2 },
  ];
  
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}>
      <Grid container spacing={3}>
        {/* User Profile Section */}
        <Grid item xs={12} md={3}>
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
        </Grid>
        
        {/* Main Content Area */}
        <Grid item xs={12} md={9}>
          <Paper elevation={2} sx={{ mb: 3 }}>
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
            </TabPanel>
            
            {/* Shift History Tab Panel */}
            <TabPanel value={activeTab} index={1}>
              <Box>
                <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
                  <Typography variant="subtitle1" fontWeight="medium">Shift History</Typography>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <Select
                        defaultValue="may"
                        size="small"
                      >
                        <MenuItem value="may">May 2025</MenuItem>
                        <MenuItem value="april">April 2025</MenuItem>
                        <MenuItem value="march">March 2025</MenuItem>
                      </Select>
                    </FormControl>
                    <Button variant="outlined" size="small" color="primary">Export</Button>
                  </Box>
                </Box>
                
                <TableContainer>
                  <Table sx={{ minWidth: 650 }} size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Date</TableCell>
                        <TableCell>Shift Start</TableCell>
                        <TableCell>Shift End</TableCell>
                        <TableCell>Total Hours</TableCell>
                        <TableCell>Station</TableCell>
                        <TableCell>Inspected</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {[
                        { date: "07/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 278, status: "Completed" },
                        { date: "06/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 295, status: "Completed" },
                        { date: "05/05/2025", start: "07:45", end: "15:30", hours: "07h 45min", station: "Station 2", inspected: 265, status: "Completed" },
                        { date: "04/05/2025", start: "08:15", end: "16:30", hours: "08h 15min", station: "Station 1", inspected: 302, status: "Completed" },
                        { date: "03/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 287, status: "Completed" },
                        { date: "02/05/2025", start: "07:30", end: "15:45", hours: "08h 15min", station: "Station 2", inspected: 310, status: "Completed" },
                        { date: "01/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 292, status: "Completed" },
                      ].map((shift, index) => (
                        <TableRow key={index} hover>
                          <TableCell>{shift.date}</TableCell>
                          <TableCell>{shift.start}</TableCell>
                          <TableCell>{shift.end}</TableCell>
                          <TableCell>{shift.hours}</TableCell>
                          <TableCell>{shift.station}</TableCell>
                          <TableCell>{shift.inspected}</TableCell>
                          <TableCell>
                            <Chip
                              label={shift.status}
                              color="success"
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <TablePagination
                  component="div"
                  count={12}
                  page={0}
                  onPageChange={() => {}}
                  rowsPerPage={7}
                  onRowsPerPageChange={() => {}}
                  rowsPerPageOptions={[7, 14, 21]}
                />
              </Box>
            </TabPanel>
            
            {/* Activities Tab Panel */}
            <TabPanel value={activeTab} index={2}>
              <Box>
                <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
                  <Typography variant="subtitle1" fontWeight="medium">Activity Log</Typography>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl size="small" sx={{ minWidth: 150 }}>
                      <Select
                        defaultValue="all"
                        size="small"
                      >
                        <MenuItem value="all">All Activities</MenuItem>
                        <MenuItem value="inspected">Inspected</MenuItem>
                        <MenuItem value="flagged">Flagged</MenuItem>
                        <MenuItem value="verified">Verified</MenuItem>
                      </Select>
                    </FormControl>
                    <Button variant="outlined" size="small" color="primary">Filter</Button>
                  </Box>
                </Box>
                
                <TableContainer>
                  <Table sx={{ minWidth: 650 }} size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Action</TableCell>
                        <TableCell>Piece Ref</TableCell>
                        <TableCell>Lot</TableCell>
                        <TableCell>Camera</TableCell>
                        <TableCell>Detection</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {[
                        { time: "14:23", action: "Inspected", pieceRef: "D532.31953.010.10", lot: "L7841", camera: "Top", detection: "OK", confidence: "98%" },
                        { time: "14:00", action: "Flagged", pieceRef: "D532.31953.012.10", lot: "L7842", camera: "Side", detection: "Wrong Lot", confidence: "95%" },
                        { time: "13:45", action: "Flagged", pieceRef: "D532.31954.010.10", lot: "L7841", camera: "Bottom", detection: "Defective", confidence: "89%" },
                        { time: "13:30", action: "Inspected", pieceRef: "D532.31953.011.10", lot: "L7841", camera: "Top", detection: "OK", confidence: "97%" },
                        { time: "13:15", action: "Inspected", pieceRef: "D532.31953.010.11", lot: "L7841", camera: "Side", detection: "OK", confidence: "96%" },
                        { time: "12:50", action: "Flagged", pieceRef: "D532.31955.010.10", lot: "L7843", camera: "Top", detection: "Missing Part", confidence: "93%" },
                        { time: "12:30", action: "Inspected", pieceRef: "D532.31953.013.10", lot: "L7841", camera: "Bottom", detection: "OK", confidence: "99%" },
                      ].map((activity, index) => (
                        <TableRow key={index} hover>
                          <TableCell>{activity.time}</TableCell>
                          <TableCell>
                            <Chip
                              label={activity.action}
                              color={
                                activity.action === "Inspected" ? "primary" : 
                                activity.action === "Verified" ? "success" : 
                                "error"
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{activity.pieceRef}</TableCell>
                          <TableCell>{activity.lot}</TableCell>
                          <TableCell>{activity.camera}</TableCell>
                          <TableCell>
                            <Chip
                              label={`${activity.detection} (${activity.confidence})`}
                              color={
                                activity.detection === "OK" ? "success" : 
                                activity.detection === "Wrong Lot" ? "warning" : 
                                "error"
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <IconButton size="small" color="primary">
                              <EyeIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" color="default">
                              <FileSearchIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <TablePagination
                  component="div"
                  count={278}
                  page={0}
                  onPageChange={() => {}}
                  rowsPerPage={7}
                  onRowsPerPageChange={() => {}}
                  rowsPerPageOptions={[7, 14, 21]}
                />
              </Box>
            </TabPanel>
            
            {/* Alerts Tab Panel */}
            <TabPanel value={activeTab} index={3}>
              <Box>
                <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
                  <Typography variant="subtitle1" fontWeight="medium">Alerts & Notifications</Typography>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <Select
                        defaultValue="all"
                        size="small"
                      >
                        <MenuItem value="all">All Alerts</MenuItem>
                        <MenuItem value="critical">Critical</MenuItem>
                        <MenuItem value="warning">Warning</MenuItem>
                        <MenuItem value="info">Info</MenuItem>
                      </Select>
                    </FormControl>
                    <Button variant="outlined" size="small" color="primary">Filter</Button>
                  </Box>
                </Box>
                
                <List component="div" sx={{ p: 0 }}>
                  {[
                    { 
                      id: "ALT-1523", 
                      type: "Critical", 
                      message: "Wrong lot detected for part D532.31954.010.10", 
                      time: "14:05", 
                      date: "07/05/2025",
                      status: "Unresolved" 
                    },
                    { 
                      id: "ALT-1522", 
                      type: "Warning", 
                      message: "Detection confidence below threshold (78%) for part D532.31953.015.10", 
                      time: "13:42", 
                      date: "07/05/2025",
                      status: "Unresolved" 
                    },
                    { 
                      id: "ALT-1521", 
                      type: "Critical", 
                      message: "Defect detected on part D532.31953.012.10", 
                      time: "11:23", 
                      date: "07/05/2025",
                      status: "Unresolved" 
                    },
                    { 
                      id: "ALT-1520", 
                      type: "Info", 
                      message: "System maintenance scheduled for 08/05/2025 22:00", 
                      time: "09:30", 
                      date: "07/05/2025",
                      status: "Acknowledged" 
                    },
                    { 
                      id: "ALT-1519", 
                      type: "Warning", 
                      message: "Camera 2 calibration may need adjustment", 
                      time: "16:15", 
                      date: "06/05/2025",
                      status: "Resolved" 
                    },
                  ].map((alert, index) => (
                    <React.Fragment key={index}>
                      <Box sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Chip
                              label={alert.type}
                              color={
                                alert.type === "Critical" ? "error" : 
                                alert.type === "Warning" ? "warning" : 
                                "info"
                              }
                              size="small"
                              sx={{ mr: 2 }}
                            />
                            <Typography variant="body2" color="text.secondary">{alert.id}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                              {alert.date} {alert.time}
                            </Typography>
                            <Chip
                              label={alert.status}
                              color={
                                alert.status === "Resolved" ? "success" : 
                                alert.status === "Acknowledged" ? "primary" : 
                                "error"
                              }
                              variant="outlined"
                              size="small"
                            />
                          </Box>
                        </Box>
                        <Typography variant="body1">{alert.message}</Typography>
                      </Box>
                      {index < 4 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </Box>
            </TabPanel>
            
 
{/* Performance Tab Panel */}
                <TabPanel value={activeTab} index={4}>
                <Box sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="h6">Performance Metrics</Typography>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <FormControl size="small" sx={{ minWidth: 150 }}>
                        <Select
                            defaultValue="thisWeek"
                            size="small"
                        >
                            <MenuItem value="thisWeek">This Week</MenuItem>
                            <MenuItem value="lastWeek">Last Week</MenuItem>
                            <MenuItem value="thisMonth">This Month</MenuItem>
                            <MenuItem value="last30Days">Last 30 Days</MenuItem>
                        </Select>
                        </FormControl>
                        <Button variant="outlined" size="small" startIcon={<FileTextIcon />}>Export Report</Button>
                    </Box>
                    </Box>
                    
                    {/* Performance Summary Cards */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={12} sm={6} md={3}>
                        <Paper sx={{ p: 2, bgcolor: 'primary.light' }}>
                        <Box sx={{ color: 'primary.main', mb: 1 }}>
                            <CameraIcon />
                        </Box>
                        <Typography variant="h4" fontWeight="bold">1,842</Typography>
                        <Typography variant="body2" color="text.secondary">Total Pieces Inspected</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Paper sx={{ p: 2, bgcolor: 'success.light' }}>
                        <Box sx={{ color: 'success.main', mb: 1 }}>
                            <CheckCircleIcon />
                        </Box>
                        <Typography variant="h4" fontWeight="bold">1,798</Typography>
                        <Typography variant="body2" color="text.secondary">Verified Correct</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
                        <Box sx={{ color: 'error.main', mb: 1 }}>
                            <AlertTriangleIcon />
                        </Box>
                        <Typography variant="h4" fontWeight="bold">44</Typography>
                        <Typography variant="body2" color="text.secondary">Issues Detected</Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <Paper sx={{ p: 2, bgcolor: 'secondary.light' }}>
                        <Box sx={{ color: 'secondary.main', mb: 1 }}>
                            <ActivityIcon />
                        </Box>
                        <Typography variant="h4" fontWeight="bold">97.6%</Typography>
                        <Typography variant="body2" color="text.secondary">Detection Accuracy</Typography>
                        </Paper>
                    </Grid>
                    </Grid>
                    
                    {/* Weekly Activity Chart - Already implemented */}
                    <Grid container spacing={3}>
                    <Grid item xs={12} md={8}>
                        <Paper elevation={1} sx={{ p: 2, height: 350 }}>
                        <Typography variant="subtitle1" gutterBottom>Weekly Activity</Typography>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart
                            data={performanceData}
                            margin={{
                                top: 5,
                                right: 30,
                                left: 20,
                                bottom: 5,
                            }}
                            >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="day" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="inspected" stroke="#2196f3" activeDot={{ r: 8 }} />
                            <Line type="monotone" dataKey="issues" stroke="#f44336" />
                            </LineChart>
                        </ResponsiveContainer>
                        </Paper>
                    </Grid>
                    
                    {/* Issues Breakdown */}
                    <Grid item xs={12} md={4}>
                        <Paper elevation={1} sx={{ p: 2, height: 350 }}>
                        <Typography variant="subtitle1" gutterBottom>Issues Breakdown</Typography>
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" fontWeight="medium" sx={{ mb: 1 }}>Issue Types</Typography>
                            {[
                            { type: "Wrong Lot", count: 18, percentage: 41 },
                            { type: "Defective Part", count: 12, percentage: 27 },
                            { type: "Missing Component", count: 8, percentage: 18 },
                            { type: "Orientation Error", count: 6, percentage: 14 },
                            ].map((issue, index) => (
                            <Box key={index} sx={{ mb: 1.5 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2">{issue.type}</Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {issue.count} ({issue.percentage}%)
                                </Typography>
                                </Box>
                                <Box sx={{ width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 8 }}>
                                <Box 
                                    sx={{ 
                                    width: `${issue.percentage}%`, 
                                    bgcolor: 'primary.main', 
                                    borderRadius: 1, 
                                    height: '100%' 
                                    }} 
                                />
                                </Box>
                            </Box>
                            ))}
                        </Box>
                        </Paper>
                    </Grid>
                    </Grid>
                    
                    {/* Camera Detection Performance */}
                    <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>Detection by Camera</Typography>
                    <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <Paper elevation={1} sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>Camera Detection</Typography>
                        {[
                            { camera: "Top Camera", count: 16, percentage: 36 },
                            { camera: "Side Camera 1", count: 14, percentage: 32 },
                            { camera: "Side Camera 2", count: 10, percentage: 23 },
                            { camera: "Bottom Camera", count: 4, percentage: 9 },
                        ].map((camera, index) => (
                            <Box key={index} sx={{ mb: 1.5 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2">{camera.camera}</Typography>
                                <Typography variant="body2" color="text.secondary">
                                {camera.count} ({camera.percentage}%)
                                </Typography>
                            </Box>
                            <Box sx={{ width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 8 }}>
                                <Box 
                                sx={{ 
                                    width: `${camera.percentage}%`, 
                                    bgcolor: 'success.main', 
                                    borderRadius: 1, 
                                    height: '100%' 
                                }} 
                                />
                            </Box>
                            </Box>
                        ))}
                        </Paper>
                    </Grid>
                    
                    {/* Efficiency Trends */}
                    <Grid item xs={12} md={6}>
                        <Paper elevation={1} sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>Efficiency Trends</Typography>
                        <Grid container spacing={2} sx={{ mt: 1 }}>
                            <Grid item xs={12} md={6}>
                            <Box sx={{ textAlign: 'center', p: 2 }}>
                                <Typography variant="h6" fontWeight="medium" color="text.secondary">Average Inspection Time</Typography>
                                <Typography variant="h3" color="primary.main" fontWeight="bold" sx={{ my: 1 }}>13.2s</Typography>
                                <Typography variant="body2" color="text.secondary">per piece</Typography>
                                <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>
                                ↓ 0.8s from last week
                                </Typography>
                            </Box>
                            </Grid>
                            <Grid item xs={12} md={6}>
                            <Box sx={{ textAlign: 'center', p: 2 }}>
                                <Typography variant="h6" fontWeight="medium" color="text.secondary">Efficiency Rating</Typography>
                                <Typography variant="h3" color="success.main" fontWeight="bold" sx={{ my: 1 }}>A+</Typography>
                                <Typography variant="body2" color="text.secondary">performance grade</Typography>
                                <Typography variant="body2" color="primary.main" sx={{ mt: 1 }}>
                                Top 5% of technicians
                                </Typography>
                            </Box>
                            </Grid>
                        </Grid>
                        </Paper>
                    </Grid>
                    </Grid>
                </Box>
                </TabPanel>

                {/* Settings Tab Panel */}
                <TabPanel value={activeTab} index={5}>
                <Box sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="h6">Account Settings</Typography>
                    <Button 
                        variant="contained" 
                        color="primary"
                        startIcon={<CheckCircleIcon />}
                    >
                        Save Changes
                    </Button>
                    </Box>
                    
                    {/* Security Settings */}
                    <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Security</Typography>
                    <Grid container spacing={3} sx={{ mt: 1 }}>
                        <Grid item xs={12} md={4}>
                        <TextField 
                            fullWidth 
                            label="Current Password" 
                            type="password" 
                            variant="outlined"
                            placeholder="Enter current password"
                        />
                        </Grid>
                        <Grid item xs={12} md={4}>
                        <TextField 
                            fullWidth 
                            label="New Password" 
                            type="password" 
                            variant="outlined"
                            placeholder="Enter new password"
                        />
                        </Grid>
                        <Grid item xs={12} md={4}>
                        <TextField 
                            fullWidth 
                            label="Confirm Password" 
                            type="password" 
                            variant="outlined"
                            placeholder="Confirm new password"
                        />
                        </Grid>
                        <Grid item xs={12}>
                        <Button 
                            variant="contained" 
                            color="primary"
                            startIcon={<ShieldIcon />}
                        >
                            Update Password
                        </Button>
                        </Grid>
                    </Grid>
                    </Paper>
                    
                    {/* Notification Preferences */}
                    <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Notification Preferences</Typography>
                    <List sx={{ width: '100%' }}>
                        <ListItem 
                        secondaryAction={
                            <Switch defaultChecked edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="Shift Reminders" 
                            secondary="Receive notifications before your shift begins" 
                        />
                        </ListItem>
                        <Divider variant="inset" component="li" />
                        
                        <ListItem 
                        secondaryAction={
                            <Switch defaultChecked edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="Critical Alerts" 
                            secondary="Receive immediate notification of critical issues" 
                        />
                        </ListItem>
                        <Divider variant="inset" component="li" />
                        
                        <ListItem 
                        secondaryAction={
                            <Switch defaultChecked edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="Performance Reports" 
                            secondary="Receive weekly performance summary" 
                        />
                        </ListItem>
                        <Divider variant="inset" component="li" />
                        
                        <ListItem 
                        secondaryAction={
                            <Switch edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="System Updates" 
                            secondary="Receive notifications about system maintenance" 
                        />
                        </ListItem>
                    </List>
                    </Paper>
                    
                    {/* Display Settings */}
                    <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Display Settings</Typography>
                    <List sx={{ width: '100%' }}>
                        <ListItem 
                        secondaryAction={
                            <Switch edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="Dark Mode" 
                            secondary="Use dark theme for interface" 
                        />
                        </ListItem>
                        <Divider variant="inset" component="li" />
                        
                        <ListItem 
                        secondaryAction={
                            <Switch defaultChecked edge="end" />
                        }
                        >
                        <ListItemText 
                            primary="Compact View" 
                            secondary="Show more content with less spacing" 
                        />
                        </ListItem>
                    </List>
                    
                    <Box sx={{ mt: 2 }}>
                        <FormControl fullWidth>
                        <InputLabel id="default-tab-label">Default Tab</InputLabel>
                        <Select
                            labelId="default-tab-label"
                            id="default-tab"
                            value="profile"
                            label="Default Tab"
                        >
                            <MenuItem value="profile">Profile</MenuItem>
                            <MenuItem value="shiftHistory">Shift History</MenuItem>
                            <MenuItem value="activities">Activities</MenuItem>
                            <MenuItem value="alerts">Alerts</MenuItem>
                            <MenuItem value="performance">Performance</MenuItem>
                        </Select>
                        </FormControl>
                    </Box>
                    </Paper>
                    
                    {/* Account Actions */}
                    <Paper elevation={1} sx={{ p: 3 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>Account Actions</Typography>
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 2 }}>
                        <Button 
                        variant="outlined" 
                        color="warning"
                        startIcon={<UsersIcon />}
                        sx={{ px: 2 }}
                        >
                        Request Access Level Change
                        </Button>
                        <Button 
                        variant="outlined" 
                        color="error"
                        startIcon={<LogOutIcon />}
                        sx={{ px: 2 }}
                        >
                        Log Out From All Devices
                        </Button>
                    </Stack>
                    </Paper>
                </Box>
                </TabPanel>
            </Paper>
            </Grid>
            </Grid>
            </Box>
  );
}
