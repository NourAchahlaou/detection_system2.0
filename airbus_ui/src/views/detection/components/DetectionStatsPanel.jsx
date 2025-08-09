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
  BarChart,
  ChevronLeft,
  ChevronRight
} from '@mui/icons-material';
import { PieChart as RechartsPieChart, Pie, Cell, ResponsiveContainer, BarChart as RechartsBarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Legend } from 'recharts';

// Import your detection statistics service
import { detectionStatisticsService } from '../service/statistics/DetectionStatisticsService';

const COLORS = {
  correct: '#4caf50',
  misplaced: '#ff9800',
  pending: '#2196f3',
  error: '#f44336',
  completed: '#8bc34a'
};

const PIE_COLORS = [COLORS.correct, COLORS.misplaced];

// Toggle Button Component for Left Side
const StatsToggleButton = ({ isOpen, onClick, hasData, isLoading }) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        right: isOpen ? 320 : 0, // Adjust based on panel width
        top: '30%',
        transform: 'translateY(-50%)',
        zIndex: 1300,
        transition: 'right 0.3s ease-in-out'
      }}
    >
      <IconButton
        onClick={onClick}
        sx={{
          bgcolor: 'secondary.main',
          color: 'white',
          width: 32,
          height: 48,
          borderRadius: '8px 0 0 8px',
          '&:hover': {
            bgcolor: 'secondary.dark',
          },
          position: 'relative',
          boxShadow: 2,
          // Add notification dot when there's data
          '&::after': hasData && !isLoading ? {
            content: '""',
            position: 'absolute',
            top: 4,
            right: 4,
            width: 8,
            height: 8,
            bgcolor: 'success.main',
            borderRadius: '50%',
            border: '1px solid white'
          } : {}
        }}
      >
        {isOpen ? <ChevronRight fontSize="small" /> : <ChevronLeft fontSize="small" />}
      </IconButton>
    </Box>
  );
};

// Collapsible Section Component
const CollapsibleSection = ({ 
  title, 
  children, 
  defaultExpanded = false, 
  icon,
  badge,
  badgeColor = "primary"
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          py: 1,
          '&:hover': {
            bgcolor: 'action.hover',
            borderRadius: 1
          }
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          {icon}
          <Typography variant="subtitle2" color="textSecondary">
            {title}
          </Typography>
          {badge && (
            <Chip
              label={badge}
              size="small"
              color={badgeColor}
              sx={{ height: 18, fontSize: '0.7rem' }}
            />
          )}
        </Stack>
        <IconButton size="small" sx={{ p: 0.5 }}>
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <Box sx={{ pl: 1, pb: 1 }}>
          {children}
        </Box>
      </Collapse>
    </Box>
  );
};

const DetectionStatsPanel = ({ 
  isOpen = false,
  onToggle,
  refreshInterval = 30000 // Increased to 30s due to service cache
}) => {
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = typeof isOpen === 'boolean' && typeof onToggle === 'function';
  const panelOpen = isControlled ? isOpen : internalOpen;

  // Use external state if provided, otherwise use internal state
  const togglePanel = () => {
    if (isControlled) {
      onToggle(!isOpen); // Pass next value
    } else {
      setInternalOpen(prev => !prev);
    }
  };

  // Auto refresh effect
  useEffect(() => {
    let interval;
    if (autoRefresh && panelOpen) {
      interval = setInterval(fetchStats, refreshInterval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, panelOpen, refreshInterval]);

  // Initial load when panel opens
  useEffect(() => {
    if (panelOpen && !stats) {
      fetchStats();
    }
  }, [panelOpen]);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Use the getDashboardData convenience method to get all data in one call
      const dashboardResult = await detectionStatisticsService.getDashboardData({
        analyticsDays: 7
      });

      if (dashboardResult.success) {
        const transformedStats = transformApiDataToComponentFormat(dashboardResult.data);
        setStats(transformedStats);
      } else {
        throw new Error(dashboardResult.error || 'Failed to load dashboard data');
      }
    } catch (err) {
      console.error('Error fetching detection stats:', err);
      setError(err.message || 'Failed to load detection statistics');
    } finally {
      setLoading(false);
    }
  };

  // Transform API data to match component's expected format
  const transformApiDataToComponentFormat = (apiData) => {
    const { overview, realTime, analytics, activeLots } = apiData;

    // Transform active lots data
    const transformedActiveLots = activeLots?.map(lot => ({
      lot_id: lot.lot_id,
      lot_name: lot.lot_name || `Lot ${lot.lot_id}`,
      expected_piece_label: lot.expected_piece_label,
      expected_piece_number: lot.expected_piece_number,
      current_correct_count: lot.current_correct_count,
      current_misplaced_count: lot.current_misplaced_count,
      total_detected: lot.total_detected,
      completion_percentage: lot.completion_percentage,
      is_target_match: lot.is_target_match,
      last_detection_time: lot.last_detection_time,
      detection_rate: lot.detection_rate || (lot.current_correct_count / lot.total_detected),
      sessions_count: lot.sessions_count || 0
    })) || [];

    // Calculate overall stats
    const totalCorrect = overview?.total_correct_pieces || 0;
    const totalMisplaced = overview?.total_misplaced_pieces || 0;
    const totalPieces = totalCorrect + totalMisplaced;
    const overallAccuracy = totalPieces > 0 ? Math.round((totalCorrect / totalPieces) * 100 * 10) / 10 : 0;

    const overallStats = {
      total_lots: overview?.total_lots || 0,
      completed_lots: overview?.completed_lots || 0,
      active_lots: overview?.active_lots || activeLots?.length || 0,
      pending_lots: overview?.pending_lots || 0,
      total_pieces_detected: totalPieces,
      total_correct_pieces: totalCorrect,
      total_misplaced_pieces: totalMisplaced,
      overall_accuracy: overallAccuracy,
      average_detection_rate: overview?.average_detection_rate || 0
    };

    // Transform recent detections
    const recentDetections = realTime?.recent_detections?.map(detection => ({
      piece_label: detection.piece_label,
      detected_count: detection.detected_count || 1,
      timestamp: new Date(detection.timestamp).toLocaleTimeString()
    })) || [];

    return {
      activeLots: transformedActiveLots,
      overallStats,
      recentDetections,
      fromCache: apiData.fromCache || false,
      timestamp: new Date().toISOString()
    };
  };

  const calculatePieData = (lot) => {
    const total = lot.total_detected || (lot.current_correct_count + lot.current_misplaced_count);
    if (total === 0) return [];
    
    return [
      { 
        name: 'Correct Pieces', 
        value: lot.current_correct_count, 
        percentage: Math.round((lot.current_correct_count / total) * 100) 
      },
      { 
        name: 'Misplaced Pieces', 
        value: lot.current_misplaced_count, 
        percentage: Math.round((lot.current_misplaced_count / total) * 100) 
      }
    ];
  };

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
  const overallPieData = stats ? [
    { 
      name: 'Correct Pieces', 
      value: stats.overallStats.total_correct_pieces,
      percentage: Math.round((stats.overallStats.total_correct_pieces / stats.overallStats.total_pieces_detected) * 100) || 0
    },
    { 
      name: 'Misplaced Pieces', 
      value: stats.overallStats.total_misplaced_pieces,
      percentage: Math.round((stats.overallStats.total_misplaced_pieces / stats.overallStats.total_pieces_detected) * 100) || 0
    }
  ] : [];

  return (
    <>
      {/* Toggle Button */}
      <StatsToggleButton 
        isOpen={panelOpen} 
        onClick={togglePanel}
        hasData={!!stats}
        isLoading={loading}
      />

      {/* Sliding Panel */}
      <Box
        sx={{
          position: 'fixed',
          right: panelOpen ? 0 : -320,
          top: 0,
          width: 320,
          height: '100vh',
          zIndex: 1200,
          transition: 'right 0.3s ease-in-out',
          bgcolor: 'background.default',
          borderRight: panelOpen ? '1px solid' : 'none',
          borderColor: 'divider',
          boxShadow: panelOpen ? 3 : 0,
          overflowY: 'auto'
        }}
      >
        <Card sx={{ height: '100%', borderRadius: 0, boxShadow: 'none' }}>
          {/* Panel Header */}
          <CardHeader
            avatar={
              <Badge badgeContent={stats?.activeLots?.length || 0} color="primary">
                <Analytics color="primary" />
              </Badge>
            }
            title={
              <Typography variant="h6" sx={{ fontSize: '1rem' }}>
                Detection Statistics
              </Typography>
            }
            action={
              <IconButton size="small" onClick={togglePanel}>
                <ChevronRight />
              </IconButton>
            }
            sx={{ pb: 1 }}
          />

          <CardContent sx={{ pt: 0, pb: 2 }}>
            {/* Header Controls */}
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
              {stats && (
                <Chip
                  size="small"
                  label={`${stats.overallStats.overall_accuracy}% Accuracy`}
                  color={stats.overallStats.overall_accuracy >= 90 ? 'success' : 'warning'}
                  sx={{ fontWeight: 'bold', fontSize: '0.7rem' }}
                />
              )}
              {stats?.fromCache && (
                <Chip
                  size="small"
                  label="Cached"
                  variant="outlined"
                  color="info"
                  sx={{ fontSize: '0.7rem' }}
                />
              )}
              <Box sx={{ flexGrow: 1 }} />
              <Tooltip title={autoRefresh ? "Pause auto-refresh" : "Resume auto-refresh"}>
                <IconButton 
                  size="small" 
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  color={autoRefresh ? "primary" : "default"}
                >
                  {autoRefresh ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh now">
                <IconButton size="small" onClick={fetchStats} disabled={loading}>
                  {loading ? <CircularProgress size={16} /> : <Refresh fontSize="small" />}
                </IconButton>
              </Tooltip>
            </Stack>

            {/* Error State */}
            {error && (
              <Alert severity="error" sx={{ mb: 2, fontSize: '0.8rem' }} action={
                <Button size="small" onClick={fetchStats}>Retry</Button>
              }>
                {error}
              </Alert>
            )}

            {/* Loading State */}
            {loading && !stats && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress size={24} />
              </Box>
            )}

            {/* No Data State */}
            {!loading && !stats && !error && (
              <Alert severity="info" action={
                <Button size="small" onClick={fetchStats}>Load Data</Button>
              }>
                No detection statistics available.
              </Alert>
            )}

            {/* Stats Content */}
            {stats && (
              <Stack spacing={1}>
                {/* Overall Performance */}
                <CollapsibleSection
                  title="Overall Performance"
                  defaultExpanded={true}
                  icon={<Assessment sx={{ fontSize: 16 }} />}
                  badge={`${stats.overallStats.total_lots} lots`}
                  badgeColor="primary"
                >
                  <Stack spacing={1}>
                    {/* Quick Stats Grid */}
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 1, textAlign: 'center', bgcolor: 'success.light', color: 'success.contrastText' }}>
                          <Typography variant="h6" fontWeight="bold" sx={{ fontSize: '1rem' }}>
                            {stats.overallStats.completed_lots}
                          </Typography>
                          <Typography variant="caption">
                            Completed
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 1, textAlign: 'center', bgcolor: 'info.light', color: 'info.contrastText' }}>
                          <Typography variant="h6" fontWeight="bold" sx={{ fontSize: '1rem' }}>
                            {stats.overallStats.active_lots}
                          </Typography>
                          <Typography variant="caption">
                            Active
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 1, textAlign: 'center', bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                          <Typography variant="h6" fontWeight="bold" sx={{ fontSize: '1rem' }}>
                            {stats.overallStats.total_correct_pieces}
                          </Typography>
                          <Typography variant="caption">
                            Correct
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 1, textAlign: 'center', bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                          <Typography variant="h6" fontWeight="bold" sx={{ fontSize: '1rem' }}>
                            {stats.overallStats.total_misplaced_pieces}
                          </Typography>
                          <Typography variant="caption">
                            Misplaced
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>

                    {/* Overall Accuracy Progress */}
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption" color="textSecondary" gutterBottom>
                        Average Detection Rate: {Math.round(stats.overallStats.average_detection_rate * 100)}%
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={stats.overallStats.average_detection_rate * 100} 
                        color="primary"
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Box>

                    {/* Overall Pie Chart - Compact */}
                    {overallPieData.length > 0 && overallPieData[0].value > 0 && (
                      <Box sx={{ textAlign: 'center', mt: 1 }}>
                        <ResponsiveContainer width="100%" height={120}>
                          <RechartsPieChart>
                            <Pie
                              data={overallPieData}
                              cx="50%"
                              cy="50%"
                              innerRadius={20}
                              outerRadius={50}
                              paddingAngle={3}
                              dataKey="value"
                              label={({ percentage }) => `${percentage}%`}
                            >
                              {overallPieData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                              ))}
                            </Pie>
                            <RechartsTooltip formatter={(value, name) => [`${value} pieces`, name]} />
                          </RechartsPieChart>
                        </ResponsiveContainer>
                      </Box>
                    )}
                  </Stack>
                </CollapsibleSection>

                <Divider />

                {/* Active Lots */}
                <CollapsibleSection
                  title="Active Lots"
                  defaultExpanded={true}
                  icon={<Inventory sx={{ fontSize: 16 }} />}
                  badge={stats.activeLots.length}
                  badgeColor="info"
                >
                  {stats.activeLots.length === 0 ? (
                    <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center', py: 1 }}>
                      No active lots
                    </Typography>
                  ) : (
                    <Stack spacing={1}>
                      {stats.activeLots.map((lot) => (
                        <Paper key={lot.lot_id} variant="outlined" sx={{ p: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="subtitle2" fontWeight="bold" noWrap sx={{ fontSize: '0.8rem' }}>
                              {lot.lot_name}
                            </Typography>
                            <Chip
                              icon={getStatusIcon(lot)}
                              label={lot.is_target_match ? 'Done' : 'Active'}
                              color={getStatusColor(lot)}
                              size="small"
                              sx={{ fontSize: '0.6rem', height: 16 }}
                            />
                          </Box>
                          
                          <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1 }}>
                            {lot.expected_piece_label} ({lot.expected_piece_number} pieces)
                          </Typography>
                          
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="caption" color="textSecondary">
                              Progress: {lot.completion_percentage}%
                            </Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={lot.completion_percentage} 
                              color={getStatusColor(lot)}
                              sx={{ height: 4, borderRadius: 2, mt: 0.5 }}
                            />
                          </Box>
                          
                          <Grid container spacing={1} sx={{ mt: 0.5 }}>
                            <Grid item xs={6}>
                              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <CheckCircle sx={{ color: COLORS.correct, fontSize: 12 }} />
                                {lot.current_correct_count}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Warning sx={{ color: COLORS.misplaced, fontSize: 12 }} />
                                {lot.current_misplaced_count}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Numbers sx={{ color: 'text.secondary', fontSize: 12 }} />
                                {lot.sessions_count} sessions
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <TrendingUp sx={{ color: 'text.secondary', fontSize: 12 }} />
                                {Math.round(lot.detection_rate * 100)}%
                              </Typography>
                            </Grid>
                          </Grid>
                        </Paper>
                      ))}
                    </Stack>
                  )}
                </CollapsibleSection>

                <Divider />

                {/* Recent Detections */}
                <CollapsibleSection
                  title="Recent Detections"
                  defaultExpanded={false}
                  icon={<Category sx={{ fontSize: 16 }} />}
                  badge={stats.recentDetections.length}
                  badgeColor="success"
                >
                  {stats.recentDetections.length === 0 ? (
                    <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center', py: 1 }}>
                      No recent detections
                    </Typography>
                  ) : (
                    <Box sx={{ maxHeight: 150, overflow: 'auto' }}>
                      <List dense disablePadding>
                        {stats.recentDetections.map((detection, index) => (
                          <React.Fragment key={index}>
                            <ListItem disablePadding sx={{ py: 0.5 }}>
                              <ListItemIcon sx={{ minWidth: 24 }}>
                                <CheckCircle color="success" sx={{ fontSize: 16 }} />
                              </ListItemIcon>
                              <ListItemText
                                primary={
                                  <Typography variant="caption" fontWeight="bold">
                                    {detection.piece_label}
                                  </Typography>
                                }
                                secondary={
                                  <Typography variant="caption" color="textSecondary">
                                    {detection.detected_count} pieces • {detection.timestamp}
                                  </Typography>
                                }
                              />
                            </ListItem>
                            {index < stats.recentDetections.length - 1 && <Divider />}
                          </React.Fragment>
                        ))}
                      </List>
                    </Box>
                  )}
                </CollapsibleSection>
              </Stack>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Backdrop for mobile */}
      {panelOpen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            right: 0,
            width: '100vw',
            height: '100vh',
            bgcolor: 'rgba(0, 0, 0, 0.3)',
            zIndex: 1100,
            display: { xs: 'block', md: 'none' }
          }}
          onClick={togglePanel}
        />
      )}
    </>
  );
};

export default DetectionStatsPanel;