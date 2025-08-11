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
  Grid,
  Paper,
  Divider
} from '@mui/material';
import {
  Analytics,
  Refresh,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  Error as ErrorIcon,
  AccessTime,
  Category,
  TrendingUp,
  PlayArrow,
  BarChart
} from '@mui/icons-material';
import { PieChart } from '@mui/x-charts/PieChart';

// Import your updated detection statistics service
import { detectionStatisticsService } from '../service/statistics/DetectionStatisticsService';

// Colors for the pie charts - Green for correct, Red for incorrect
const LOT_MATCHING_COLORS = ['#4caf50', '#f44336']; // Green for correct, Red for incorrect

// Center Label Component for Pie Charts
const PieCenterLabel = ({ primaryText, secondaryText }) => (
  <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central">
    <tspan x="50%" dy="-0.3em" fontSize="24" fontWeight="bold" fill="#333">
      {primaryText}
    </tspan>
    <tspan x="50%" dy="1.2em" fontSize="14" fill="#666">
      {secondaryText}
    </tspan>
  </text>
);

// Toggle Button Component
const StatsToggleButton = ({ isOpen, onClick, hasData, isLoading, isDetectionActive }) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        right: isOpen ? 400 : 0,
        top: '30%',
        transform: 'translateY(-50%)',
        zIndex: 1300,
        transition: 'right 0.3s ease-in-out'
      }}
    >
      <IconButton
        onClick={onClick}
        sx={{
          bgcolor: 'primary.main',
          color: 'white',
          width: 32,
          height: 48,
          borderRadius: '8px 0 0 8px',
          '&:hover': {
            bgcolor: 'primary.dark',
          },
          position: 'relative',
          boxShadow: 2,
          '&::after': hasData && isDetectionActive && !isLoading ? {
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

// No Detection State Component
const NoDetectionState = ({ onStartDetection }) => (
  <Box sx={{ textAlign: 'center', py: 4 }}>
    <BarChart sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
    <Typography variant="h6" color="text.secondary" gutterBottom>
      No Detection Statistics Available
    </Typography>
    <Typography variant="body2" color="text.disabled" sx={{ mb: 3, maxWidth: 300, mx: 'auto' }}>
      Detection statistics will be available once you start running detection on your selected lot.
    </Typography>
    <Alert severity="info" sx={{ mb: 2 }}>
      <Typography variant="body2">
        <strong>Start Detection</strong> to see real-time statistics about your current lot matching and session details.
      </Typography>
    </Alert>
    {onStartDetection && (
      <Button
        variant="contained"
        startIcon={<PlayArrow />}
        onClick={onStartDetection}
        sx={{ mt: 1 }}
      >
        Start Detection
      </Button>
    )}
  </Box>
);

const DetectionStatsPanel = ({ 
  isOpen = false,
  onToggle,
  refreshInterval = 30000,
  isDetectionActive = false,
  onStartDetection = null,
  currentLotId = null // NEW: Current lot ID prop
}) => {
  const [loading, setLoading] = useState(false);
  const [lotSummary, setLotSummary] = useState(null);
  const [lastSessions, setLastSessions] = useState([]);
  const [error, setError] = useState(null);
  const [internalOpen, setInternalOpen] = useState(false);
  const [hasEverDetected, setHasEverDetected] = useState(false);
  
  const isControlled = typeof isOpen === 'boolean' && typeof onToggle === 'function';
  const panelOpen = isControlled ? isOpen : internalOpen;

  const togglePanel = () => {
    if (isControlled) {
      onToggle(!isOpen);
    } else {
      setInternalOpen(prev => !prev);
    }
  };

  // Auto refresh effect - only when detection is active
  useEffect(() => {
    let interval;
    if (panelOpen && isDetectionActive && currentLotId) {
      interval = setInterval(fetchAllData, refreshInterval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [panelOpen, refreshInterval, isDetectionActive, currentLotId]);

  // Initial load when panel opens and detection is active
  useEffect(() => {
    if (panelOpen && isDetectionActive && currentLotId) {
      fetchAllData();
    }
  }, [panelOpen, isDetectionActive, currentLotId]);

  // Track if detection has ever been active
  useEffect(() => {
    if (isDetectionActive) {
      setHasEverDetected(true);
    }
  }, [isDetectionActive]);

  const fetchAllData = async () => {
    // Only fetch data if detection is active and we have a lot ID
    if (!isDetectionActive || !currentLotId) {
      console.log('🔍 Skipping stats fetch - detection not active or no lot ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Fetch lot summary for the current lot
      const summaryResult = await detectionStatisticsService.getLotSummary(currentLotId);
      if (summaryResult.success) {
        setLotSummary(summaryResult.data);
        console.log("lotsummery :)",summaryResult)
      } else if (summaryResult.notFound) {
        setError(`Lot ${currentLotId} not found`);
        return;
      } else {
        console.warn('Failed to fetch lot summary:', summaryResult.error);
      }

      // Fetch last sessions per lot to get current lot's session
      const sessionsResult = await detectionStatisticsService.getLastSessionsPerLot();
      if (sessionsResult.success) {
        // Filter for current lot only
        const currentLotSessions = sessionsResult.data.filter(session => 
          session.lot_id === currentLotId
        );
        console.log("currentLotSessions",currentLotSessions,"sessionsResult",sessionsResult)
        setLastSessions(currentLotSessions);
      } else {
        console.warn('Failed to fetch last sessions:', sessionsResult.error);
      }

    } catch (err) {
      console.error('Error fetching detection stats:', err);
      setError(err.message || 'Failed to load detection statistics');
    } finally {
      setLoading(false);
    }
  };

  // Transform lot summary data for pie chart (correct vs incorrect pieces)
  const getLotMatchingChartData = () => {
    if (!lotSummary) return [];
    
    const correctPieces = lotSummary.correct_pieces_count || 0;
    const totalDetected = lotSummary.total_detections || 0;
    const incorrectPieces = totalDetected - correctPieces;
    
    return [
      {
        id: 'correct',
        value: correctPieces,
        label: 'Correct Pieces',
        color: LOT_MATCHING_COLORS[0] // Green
      },
      {
        id: 'incorrect',
        value: incorrectPieces,
        label: 'Incorrect Pieces',
        color: LOT_MATCHING_COLORS[1] // Red
      }
    ].filter(item => item.value > 0); // Only show non-zero values
  };

  // Calculate matching percentage
  const getMatchingPercentage = () => {
    if (!lotSummary || !lotSummary.total_detections) return 0;
    
    const correctPieces = lotSummary.correct_pieces_count || 0;
    const totalDetected = lotSummary.total_detections || 0;
    
    return totalDetected > 0 ? Math.round((correctPieces / totalDetected) * 100) : 0;
  };

  const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const formatDuration = (startTime, endTime) => {
    if (!startTime || !endTime) return 'N/A';
    const start = new Date(startTime);
    const end = new Date(endTime);
    const durationMs = end - start;
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const hasData = lotSummary || lastSessions.length > 0;
  const shouldShowStats = isDetectionActive && hasData && currentLotId;
  const shouldShowNoDetectionState = !isDetectionActive && !hasEverDetected;
  const shouldShowWaitingState = (!isDetectionActive && hasEverDetected) || (!currentLotId && isDetectionActive);
  const currentSession = lastSessions.length > 0 ? lastSessions[0] : null;

  return (
    <>
      {/* Toggle Button */}
      <StatsToggleButton 
        isOpen={panelOpen} 
        onClick={togglePanel}
        hasData={!!hasData}
        isLoading={loading}
        isDetectionActive={isDetectionActive}
      />

      {/* Sliding Panel */}
      <Box
        sx={{
          position: 'fixed',
          right: panelOpen ? 0 : -400,
          top: 0,
          width: 400,
          height: '100vh',
          zIndex: 1200,
          transition: 'right 0.3s ease-in-out',
          bgcolor: 'background.default',
          borderLeft: panelOpen ? '1px solid' : 'none',
          borderColor: 'divider',
          boxShadow: panelOpen ? 3 : 0,
          overflowY: 'auto'
        }}
      >
        <Card sx={{ height: '100%', borderRadius: 0, boxShadow: 'none' }}>
          {/* Panel Header */}
          <CardHeader
            avatar={<Analytics color="primary" />}
            title={
              <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
                Lot Detection Statistics
                {isDetectionActive && (
                  <Chip 
                    label="LIVE" 
                    size="small" 
                    color="success" 
                    sx={{ ml: 1, fontSize: '0.75rem' }}
                  />
                )}
              </Typography>
            }
            subheader={
              currentLotId && lotSummary && (
                <Typography variant="body2" color="text.secondary">
                  {lotSummary.lot_name} (ID: {currentLotId})
                </Typography>
              )
            }
            action={
              <Stack direction="row" spacing={1}>
                {isDetectionActive && currentLotId && (
                  <Tooltip title="Refresh data">
                    <IconButton size="small" onClick={fetchAllData} disabled={loading}>
                      {loading ? <CircularProgress size={16} /> : <Refresh fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                )}
                <IconButton size="small" onClick={togglePanel}>
                  <ChevronRight />
                </IconButton>
              </Stack>
            }
            sx={{ pb: 1 }}
          />

          <CardContent sx={{ pt: 0, pb: 2 }}>
            {/* No Detection State */}
            {shouldShowNoDetectionState && (
              <NoDetectionState onStartDetection={onStartDetection} />
            )}

            {/* Waiting for Detection State */}
            {shouldShowWaitingState && (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <AccessTime sx={{ fontSize: 60, color: 'warning.main', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {!currentLotId ? 'No Lot Selected' : 'Detection Stopped'}
                </Typography>
                <Typography variant="body2" color="text.disabled" sx={{ mb: 3 }}>
                  {!currentLotId 
                    ? 'Select a lot and start detection to see statistics.'
                    : 'Statistics are paused. Start detection again to see live updates.'
                  }
                </Typography>
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Lot-specific statistics are only updated during active detection sessions.
                  </Typography>
                </Alert>
                {onStartDetection && (
                  <Button
                    variant="outlined"
                    startIcon={<PlayArrow />}
                    onClick={onStartDetection}
                    sx={{ mt: 1 }}
                  >
                    {!currentLotId ? 'Start Detection' : 'Resume Detection'}
                  </Button>
                )}
              </Box>
            )}

            {/* Error State */}
            {error && isDetectionActive && (
              <Alert severity="error" sx={{ mb: 2 }} action={
                <Button size="small" onClick={fetchAllData}>Retry</Button>
              }>
                {error}
              </Alert>
            )}

            {/* Loading State */}
            {loading && isDetectionActive && !hasData && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            )}

            {/* No Data State (during active detection) */}
            {!loading && !hasData && !error && isDetectionActive && currentLotId && (
              <Alert severity="info" action={
                <Button size="small" onClick={fetchAllData}>Load Data</Button>
              }>
                No detection statistics available yet for this lot. Continue detecting to generate data.
              </Alert>
            )}

            {/* Stats Content - Only show during active detection */}
            {shouldShowStats && (
              <Stack spacing={3}>
                {/* Lot Matching Pie Chart */}
                {lotSummary && lotSummary.total_detections > 0 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                      Lot Matching Results
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Expected: {lotSummary.expected_piece_count} pieces
                    </Typography>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'center', position: 'relative' }}>
                      <PieChart
                        colors={LOT_MATCHING_COLORS}
                        margin={{ left: 80, right: 80, top: 80, bottom: 80 }}
                        series={[
                          {
                            data: getLotMatchingChartData(),
                            innerRadius: 75,
                            outerRadius: 100,
                            paddingAngle: 2,
                            highlightScope: { faded: 'global', highlighted: 'item' }
                          }
                        ]}
                        height={260}
                        width={260}
                        slotProps={{
                          legend: { hidden: true }
                        }}
                      >
                        <PieCenterLabel 
                          primaryText={`${getMatchingPercentage()}%`}
                          secondaryText="Match Rate"
                        />
                      </PieChart>
                    </Box>

                    {/* Stats Summary */}
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light', color: 'success.contrastText', borderRadius: 2 }}>
                          <Typography variant="h5" fontWeight="bold">
                            {lotSummary.correct_pieces_count || 0}
                          </Typography>
                          <Typography variant="body2">
                            Correct Pieces
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6}>
                        <Paper elevation={0} sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light', color: 'error.contrastText', borderRadius: 2 }}>
                          <Typography variant="h5" fontWeight="bold">
                            {(lotSummary.total_detections || 0) - (lotSummary.correct_pieces_count || 0)}
                          </Typography>
                          <Typography variant="body2">
                            Incorrect Pieces
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12}>
                        <Chip
                          icon={lotSummary.is_completed ? <CheckCircle /> : <AccessTime />}
                          label={lotSummary.is_completed ? 'Lot Completed' : 'Detection In Progress'}
                          color={lotSummary.is_completed ? 'success' : 'primary'}
                          sx={{ width: '100%', py: 1 }}
                        />
                      </Grid>
                    </Grid>
                  </Box>
                )}

                {/* Show message if no detections yet */}
                {lotSummary && lotSummary.total_detections === 0 && (
                  <Alert severity="info">
                    <Typography variant="body2">
                      No detections recorded yet for this lot. The pie chart will appear once detection starts.
                    </Typography>
                  </Alert>
                )}

                <Divider />

                {/* Last Session Details */}
                {currentSession && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                      Latest Session Details
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
                      <Stack spacing={2}>
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Session ID
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              #{currentSession.last_session_id}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Status
                            </Typography>
                            <Chip
                              icon={currentSession.is_target_match ? <CheckCircle /> : <AccessTime />}
                              label={currentSession.is_target_match ? 'Completed' : 'Active'}
                              color={currentSession.is_target_match ? 'success' : 'primary'}
                              size="small"
                            />
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Accuracy Rate
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {Math.round(currentSession.detection_rate || 0)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Completion Rate
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {currentSession.total_detected
                                ? Math.round(((currentSession.correct_pieces + currentSession.misplaced_pieces) / currentSession.total_detected) * 100)
                                : 0}%
                            </Typography>

                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Total Detections
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {currentSession.total_detected || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Confidence Score
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {Math.round((currentSession.confidence_score || 0) * 100)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Started At
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {formatDateTime(currentSession.last_session_time)}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Stack>
                    </Paper>
                  </Box>
                )}

                {/* Lot Summary Information */}
                {lotSummary && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                      Lot Overview
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="textSecondary">
                            Lot Name
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {lotSummary.lot_name}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="textSecondary">
                            Expected Pieces
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {lotSummary.expected_piece_count}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="textSecondary">
                            Total Sessions
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {lotSummary.total_sessions || 0}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="textSecondary">
                            Last Updated
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {formatDateTime(lotSummary.last_session_date)}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Box>
                )}
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