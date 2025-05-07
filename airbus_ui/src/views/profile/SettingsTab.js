import * as React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Switch,
  List,
  ListItem,
  ListItemText,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ShieldIcon from '@mui/icons-material/Shield';
import LogOutIcon from '@mui/icons-material/Logout';
import UsersIcon from '@mui/icons-material/Group';

export default function SettingsTab({ activeTab }) {
  return (
    <Box sx={{ p: 3 }}>
      <Grid container>
        <Grid item xs={12} variant="outlined">
         
            {/* Settings Tab Panel */}
            
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
                      secondaryAction={<Switch defaultChecked edge="end" />}
                    >
                      <ListItemText
                        primary="Shift Reminders"
                        secondary="Receive notifications before your shift begins"
                      />
                    </ListItem>
                    <Divider variant="inset" component="li" />

                    <ListItem
                      secondaryAction={<Switch defaultChecked edge="end" />}
                    >
                      <ListItemText
                        primary="Critical Alerts"
                        secondary="Receive immediate notification of critical issues"
                      />
                    </ListItem>
                    <Divider variant="inset" component="li" />

                    <ListItem
                      secondaryAction={<Switch defaultChecked edge="end" />}
                    >
                      <ListItemText
                        primary="Performance Reports"
                        secondary="Receive weekly performance summary"
                      />
                    </ListItem>
                    <Divider variant="inset" component="li" />

                    <ListItem
                      secondaryAction={<Switch edge="end" />}
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
                      secondaryAction={<Switch edge="end" />}
                    >
                      <ListItemText
                        primary="Dark Mode"
                        secondary="Use dark theme for interface"
                      />
                    </ListItem>
                    <Divider variant="inset" component="li" />

                    <ListItem
                      secondaryAction={<Switch defaultChecked edge="end" />}
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
       
        </Grid>
      </Grid>
    </Box>
  );
}
