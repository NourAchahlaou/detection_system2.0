import * as React from 'react';
import DescriptionIcon from '@mui/icons-material/Description';
import {
  Box,
  Grid,
  Typography,
  Paper,
  FormControl,
  Select,
  MenuItem,
  Button
} from '@mui/material';
import {
  CameraAlt as CameraIcon,
  CheckCircle as CheckCircleIcon,
  Warning as AlertTriangleIcon,

  TrendingUp as ActivityIcon,
  
} from '@mui/icons-material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const performanceData = [
  { day: 'Mon', inspected: 45, issues: 3 },
  { day: 'Tue', inspected: 52, issues: 5 },
  { day: 'Wed', inspected: 49, issues: 2 },
  { day: 'Thu', inspected: 50, issues: 4 },
  { day: 'Fri', inspected: 43, issues: 3 },
  { day: 'Sat', inspected: 30, issues: 1 },
  { day: 'Today', inspected: 25, issues: 2 },
];

export default function PerformanceTab() {
  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Performance Metrics</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <Select defaultValue="thisWeek" size="small">
              <MenuItem value="thisWeek">This Week</MenuItem>
              <MenuItem value="lastWeek">Last Week</MenuItem>
              <MenuItem value="thisMonth">This Month</MenuItem>
              <MenuItem value="last30Days">Last 30 Days</MenuItem>
            </Select>
          </FormControl>
          <Button variant="outlined" size="small" startIcon={<DescriptionIcon />}>
  Export Report
</Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { icon: <CameraIcon />, value: 1842, label: 'Total Pieces Inspected', color: 'primary' },
          { icon: <CheckCircleIcon />, value: 1798, label: 'Verified Correct', color: 'success' },
          { icon: <AlertTriangleIcon />, value: 44, label: 'Issues Detected', color: 'error' },
          { icon: <ActivityIcon />, value: '97.6%', label: 'Detection Accuracy', color: 'secondary' }
        ].map((item, index) => (
          <Grid key={index} item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, bgcolor: `${item.color}.light` }}>
              <Box sx={{ color: `${item.color}.main`, mb: 1 }}>{item.icon}</Box>
              <Typography variant="h4" fontWeight="bold">{item.value}</Typography>
              <Typography variant="body2" color="text.secondary">{item.label}</Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper elevation={1} sx={{ p: 2, height: 350 }}>
            <Typography variant="subtitle1" gutterBottom>Weekly Activity</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="inspected" stroke="#2196f3" />
                <Line type="monotone" dataKey="issues" stroke="#f44336" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper elevation={1} sx={{ p: 2, height: 350 }}>
            <Typography variant="subtitle1" gutterBottom>Issues Breakdown</Typography>
            <Box sx={{ mt: 2 }}>
              {[
                { type: "Wrong Lot", count: 18, percentage: 41 },
                { type: "Defective Part", count: 12, percentage: 27 },
                { type: "Missing Component", count: 8, percentage: 18 },
                { type: "Orientation Error", count: 6, percentage: 14 },
              ].map((issue, i) => (
                <Box key={i} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">{issue.type}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {issue.count} ({issue.percentage}%)
                    </Typography>
                  </Box>
                  <Box sx={{ width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 8, mt: 0.5 }}>
                    <Box
                      sx={{
                        width: `${issue.percentage}%`,
                        bgcolor: 'primary.main',
                        height: '100%',
                        borderRadius: 1
                      }}
                    />
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
