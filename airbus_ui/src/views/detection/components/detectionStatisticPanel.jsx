import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Stack,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  CircularProgress,
  Collapse,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Badge,
  LinearProgress
} from '@mui/material';
import {
  Analytics,
  Inventory,
  CheckCircle,
  Warning,
  Error,
  TrendingUp,
  TrendingDown,
  ExpandLess,
  ExpandMore,
  Refresh,
  Info,
  PlayArrow,
  Pause,
  Category,
  Numbers,
  Assessment,
  PieChart,
  BarChart
} from '@mui/icons-material';
import { PieChart as RechartsPieChart, Pie, Cell, ResponsiveContainer, BarChart as RechartsBarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Legend } from 'recharts';

// Mock data - replace with actual API calls
const mockDetectionStats = {
  activeLots: [
    {
      lot_id: 1,
      lot_name: "Lot_2024-01-15_Widget_A",
      expected_piece_label: "Widget_A",
      expected_piece_number: 50,
      current_correct_count: 42,
      current_misplaced_count: 8,
      total_detected: 50,
      completion_percentage: 84,
      is_target_match: false,
      last_detection_time: "2024-01-15T10:30:00Z",
      detection_rate: 0.84,
      sessions_count: 15
    },
    {
      lot_id: 2,
      lot_name: "Lot_2024-01-15_Widget_B",
      expected_piece_label: "Widget_B", 
      expected_piece_number: 25,
      current_correct_count: 25,
      current_misplaced_count: 0,
      total_detected: 25,
      completion_percentage: 100,
      is_target_match: true,
      last_detection_time: "2024-01-15T11:15:00Z",
      detection_rate: 1.0,
      sessions_count: 8
    }
  ],
  overallStats: {
    total_lots: 5,
    completed_lots: 2,
    active_lots: 2,
    pending_lots: 1,
    total_pieces_detected: 1250,
    total_correct_pieces: 1180,
    total_misplaced_pieces: 70,
    overall_accuracy: 94.4,
    average_detection_rate: 0.92
  },
  recentDetections: [
    { piece_label: "Widget_A", detected_count: 5, timestamp: "10:30:15" },
    { piece_label: "Widget_B", detected_count: 3, timestamp: "10:28:42" },
    { piece_label: "Widget_C", detected_count: 2, timestamp: "10:25:18" }
  ]
};

const COLORS = {
  correct: '#4caf50',
  misplaced: '#ff9800',
  pending: '#2196f3',
  error: '#f44336',
  completed: '#8bc34a'
};

const PIE_COLORS = [COLORS.correct, COLORS.misplaced];

const DetectionStatsPanel = ({ 
  detectionService,
  isOpen = false,
  onToggle,
  refreshInterval = 5000 
}) => {
  const [expanded, setExpanded] = useState(isOpen);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(mockDetectionStats);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Auto refresh effect
  useEffect(() => {
    let interval;
    if (autoRefresh && expanded) {
      interval = setInterval(fetchStats, refreshInterval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, expanded, refreshInterval]);

  // Initial load when expanded
  useEffect(() => {
    if (expanded && !stats) {
      fetchStats();
    }
  }, [expanded]);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      // Replace with actual API calls to your detection service
      // const lotStats = await detectionService.getAllDetectionLots();
      // const overallStats = await detectionService.getOverallStats();
      
      // For now, simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update with real data when available
      setStats(mockDetectionStats);
    } catch (err) {
      console.error('Error fetching detection stats:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = () => {
    const newExpanded = !expanded;
    setExpanded(newExpanded);
    if (onToggle) {
      onToggle(newExpanded);
    }
    if (newExpanded) {
      fetchStats();
    }
  };

  const calculatePieData = (lot) => [
    { 
      name: 'Correct Pieces', 
      value: lot.current_correct_count, 
      percentage: Math.round((lot.current_correct_count / lot.total_detected) * 100) 
    },
    { 
      name: 'Misplaced Pieces', 
      value: lot.current_misplaced_count, 
      percentage: Math.round((lot.current_misplaced_count / lot.total_detected) * 100) 
    }
  ];

  const getStatusColor = (lot) => {
    if (lot.is_target_match) return 'success';
    if (lot.completion_percentage >= 80) return 'warning';
    return 'info';
  };

  const getStatusIcon = (lot) => {
    if (lot.is_target_match) return <CheckCircle />;
    if (lot.completion_percentage >= 80) return <Warning />;
    return <Info />;
  };

  // Calculate overall pie data
  const overallPieData = [
    { 
      name: 'Correct Pieces', 
      value: stats.overallStats.total_correct_pieces,
      percentage: Math.round((stats.overallStats.total_correct_pieces / stats.overallStats.total_pieces_detected) * 100)
    },
    { 
      name: 'Misplaced Pieces', 
      value: stats.overallStats.total_misplaced_pieces,
      percentage: Math.round((stats.overallStats.total_misplaced_pieces / stats.overallStats.total_pieces_detected) * 100)
    }
  ];

  return (
    <Card sx={{ width: '100%', maxWidth: 1200, mx: 'auto' }}>
      {/* Toggle Button Header */}
      <CardHeader
        avatar={
          <Badge badgeContent={stats.activeLots.length} color="primary">
            <Analytics color="primary" />
          </Badge>
        }
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" component="div">
              Detection Statistics
            </Typography>
            <Chip
              size="small"
              label={`${stats.overallStats.overall_accuracy}% Accuracy`}
              color={stats.overallStats.overall_accuracy >= 90 ? 'success' : 'warning'}
              sx={{ fontWeight: 'bold' }}
            />
          </Box>
        }
        action={
          <Stack direction="row" spacing={1} alignItems="center">
            {expanded && (
              <>
                <Tooltip title={autoRefresh ? "Pause auto-refresh" : "Resume auto-refresh"}>
                  <IconButton 
                    size="small" 
                    onClick={() => setAutoRefresh(!autoRefresh)}
                    color={autoRefresh ? "primary" : "default"}
                  >
                    {autoRefresh ? <Pause /> : <PlayArrow />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh now">
                  <IconButton size="small" onClick={fetchStats} disabled={loading}>
                    {loading ? <CircularProgress size={20} /> : <Refresh />}
                  </IconButton>
                </Tooltip>
              </>
            )}
            <Button
              variant="outlined"
              size="small"
              onClick={handleToggle}
              endIcon={expanded ? <ExpandLess /> : <ExpandMore />}
              sx={{ minWidth: 120 }}
            >
              {expanded ? 'Hide Stats' : 'Show Stats'}
            </Button>
          </Stack>
        }
        sx={{ pb: 1 }}
      />

      <Collapse in={expanded}>
        <CardContent sx={{ pt: 0 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} action={
              <Button size="small" onClick={fetchStats}>Retry</Button>
            }>
              Failed to load detection statistics: {error}
            </Alert>
          )}

          {/* Overall Statistics */}
          <Paper elevation={1} sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Assessment color="primary" />
              Overall Performance
            </Typography>
            
            <Grid container spacing={3}>
              {/* Overall Pie Chart */}
              <Grid item xs={12} md={4}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                    Overall Accuracy Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <RechartsPieChart>
                      <Pie
                        data={overallPieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, percentage }) => `${percentage}%`}
                      >
                        {overallPieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip formatter={(value, name) => [`${value} pieces`, name]} />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </Box>
              </Grid>

              {/* Stats Cards */}
              <Grid item xs={12} md={8}>
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light', color: 'success.contrastText' }}>
                      <Typography variant="h4" fontWeight="bold">
                        {stats.overallStats.completed_lots}
                      </Typography>
                      <Typography variant="body2">
                        Completed Lots
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light', color: 'info.contrastText' }}>
                      <Typography variant="h4" fontWeight="bold">
                        {stats.overallStats.active_lots}
                      </Typography>
                      <Typography variant="body2">
                        Active Lots
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                      <Typography variant="h4" fontWeight="bold">
                        {stats.overallStats.total_correct_pieces}
                      </Typography>
                      <Typography variant="body2">
                        Correct Pieces
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                      <Typography variant="h4" fontWeight="bold">
                        {stats.overallStats.total_misplaced_pieces}
                      </Typography>
                      <Typography variant="body2">
                        Misplaced Pieces
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Average Detection Rate: {Math.round(stats.overallStats.average_detection_rate * 100)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={stats.overallStats.average_detection_rate * 100} 
                    color="primary"
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              </Grid>
            </Grid>
          </Paper>

          {/* Active Lots Details */}
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Inventory color="primary" />
            Active Lots ({stats.activeLots.length})
          </Typography>

          <Grid container spacing={2}>
            {stats.activeLots.map((lot) => {
              const pieData = calculatePieData(lot);
              
              return (
                <Grid item xs={12} md={6} key={lot.lot_id}>
                  <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardHeader
                      avatar={
                        <Chip
                          icon={getStatusIcon(lot)}
                          label={lot.is_target_match ? 'Completed' : 'Active'}
                          color={getStatusColor(lot)}
                          size="small"
                        />
                      }
                      title={
                        <Typography variant="subtitle1" fontWeight="bold" noWrap>
                          {lot.lot_name}
                        </Typography>
                      }
                      subheader={
                        <Typography variant="body2" color="textSecondary">
                          Expected: {lot.expected_piece_label} ({lot.expected_piece_number} pieces)
                        </Typography>
                      }
                      sx={{ pb: 1 }}
                    />
                    
                    <CardContent sx={{ pt: 0 }}>
                      <Grid container spacing={2}>
                        {/* Pie Chart */}
                        <Grid item xs={12} sm={6}>
                          <Box sx={{ textAlign: 'center' }}>
                            <ResponsiveContainer width="100%" height={150}>
                              <RechartsPieChart>
                                <Pie
                                  data={pieData}
                                  cx="50%"
                                  cy="50%"
                                  innerRadius={25}
                                  outerRadius={60}
                                  paddingAngle={5}
                                  dataKey="value"
                                  label={({ percentage }) => `${percentage}%`}
                                >
                                  {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                                  ))}
                                </Pie>
                                <RechartsTooltip formatter={(value, name) => [`${value} pieces`, name]} />
                              </RechartsPieChart>
                            </ResponsiveContainer>
                          </Box>
                        </Grid>

                        {/* Stats */}
                        <Grid item xs={12} sm={6}>
                          <Stack spacing={1}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2" color="textSecondary">Progress</Typography>
                              <Typography variant="body2" fontWeight="bold">{lot.completion_percentage}%</Typography>
                            </Box>
                            <LinearProgress 
                              variant="determinate" 
                              value={lot.completion_percentage} 
                              color={getStatusColor(lot)}
                              sx={{ height: 6, borderRadius: 3 }}
                            />
                            
                            <Stack spacing={0.5} sx={{ mt: 1 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <CheckCircle sx={{ color: COLORS.correct, fontSize: 16 }} />
                                <Typography variant="body2">
                                  Correct: <strong>{lot.current_correct_count}</strong>
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Warning sx={{ color: COLORS.misplaced, fontSize: 16 }} />
                                <Typography variant="body2">
                                  Misplaced: <strong>{lot.current_misplaced_count}</strong>
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Numbers sx={{ color: 'text.secondary', fontSize: 16 }} />
                                <Typography variant="body2">
                                  Sessions: <strong>{lot.sessions_count}</strong>
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <TrendingUp sx={{ color: 'text.secondary', fontSize: 16 }} />
                                <Typography variant="body2">
                                  Rate: <strong>{Math.round(lot.detection_rate * 100)}%</strong>
                                </Typography>
                              </Box>
                            </Stack>
                          </Stack>
                        </Grid>
                      </Grid>
                      
                      <Divider sx={{ my: 1 }} />
                      
                      <Typography variant="caption" color="textSecondary">
                        Last detection: {new Date(lot.last_detection_time).toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>

          {/* Recent Detections */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Category color="primary" />
              Recent Detections
            </Typography>
            
            <Paper elevation={1} sx={{ maxHeight: 200, overflow: 'auto' }}>
              <List dense>
                {stats.recentDetections.map((detection, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary={detection.piece_label}
                        secondary={`${detection.detected_count} pieces detected`}
                      />
                      <Typography variant="caption" color="textSecondary">
                        {detection.timestamp}
                      </Typography>
                    </ListItem>
                    {index < stats.recentDetections.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </Paper>
          </Box>
        </CardContent>
      </Collapse>
    </Card>
  );
};

// // Example usage component
// const ExampleUsage = () => {
//   const [panelOpen, setPanelOpen] = useState(false);

//   return (
//     <Box sx={{ p: 3 }}>
//       <DetectionStatsPanel 
//         isOpen={panelOpen}
//         onToggle={setPanelOpen}
//         refreshInterval={5000}
//       />
      
//       <Box sx={{ mt: 2, textAlign: 'center' }}>
//         <Typography variant="body2" color="textSecondary">
//           This panel shows real-time detection statistics with expandable details
//         </Typography>
//       </Box>
//     </Box>
//   );
// };

// export default ExampleUsage;