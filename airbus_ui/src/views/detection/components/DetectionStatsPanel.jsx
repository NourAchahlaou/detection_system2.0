import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  AccessTime,
  PlayArrow,
  BarChart
} from '@mui/icons-material';
import { PieChart } from '@mui/x-charts/PieChart';

// Import your updated detection statistics service
import { detectionStatisticsService } from '../service/statistics/DetectionStatisticsService';

// Colors for the pie charts - Green for correct, Red for incorrect
const SESSION_MATCHING_COLORS = ['#4caf50', '#f44336']; // Green for correct, Red for incorrect

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
        <strong>Start Detection</strong> to see real-time statistics about your current session details.
      </Typography>
    </Alert>
  </Box>
);

const DetectionStatsPanel = ({ 
  isOpen = false,
  onToggle,
  isDetectionActive = false,
  onStartDetection = null,
  currentLotId = null,
  detectionCompleted = null, // Signal that a detection just completed
}) => {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [lastSessions, setLastSessions] = useState([]);
  const [error, setError] = useState(null);
  const [internalOpen, setInternalOpen] = useState(false);
  const [hasEverDetected, setHasEverDetected] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  
  // Refs for managing race conditions
  const mountedRef = useRef(true);
  const lastFetchRef = useRef(0);
  const detectionCompletedRef = useRef(null);
  
  const isControlled = typeof isOpen === 'boolean' && typeof onToggle === 'function';
  const panelOpen = isControlled ? isOpen : internalOpen;

  const togglePanel = () => {
    if (isControlled) {
      onToggle(!isOpen);
    } else {
      setInternalOpen(prev => !prev);
    }
  };

  // FIXED: Fetch function with rate limiting - NO auto-refresh
  const fetchFreshData = useCallback(async (isManualRefresh = false, bypassCache = false) => {
    // Rate limiting - prevent too frequent calls
    const now = Date.now();
    const minInterval = isManualRefresh ? 1000 : 3000; // Manual: 1s, Auto: 3s
    
    if (!bypassCache && (now - lastFetchRef.current) < minInterval) {
      console.log('🔍 Skipping fetch - rate limited');
      return;
    }
    lastFetchRef.current = now;

    if (!currentLotId) {
      console.log('🔍 Skipping stats fetch - no lot ID');
      return;
    }

    if (isManualRefresh) {
      setRefreshing(true);
    } else if (!lastSessions.length) {
      setLoading(true);
    }
    setError(null);
    
    try {
      console.log('🔍 Fetching detection stats...', { 
        isManualRefresh, 
        bypassCache,
        currentLotId, 
        isDetectionActive 
      });

      // Clear cache on manual refresh or detection completion
      if (isManualRefresh || bypassCache) {
        detectionStatisticsService.clearLocalCache();
      }

      // Fetch sessions
      const sessionsResult = await detectionStatisticsService.getLastSessionsPerLot();

      if (!mountedRef.current) return;

      // Process sessions - filter for current lot
      if (sessionsResult.success) {
        const currentLotSessions = sessionsResult.data.filter(session => 
          session.lot_id === currentLotId
        );
        console.log("✅ Sessions loaded:", currentLotSessions);
        setLastSessions(currentLotSessions);
      } else {
        console.warn('Failed to fetch last sessions:', sessionsResult.error);
      }

      setLastUpdateTime(new Date().toLocaleTimeString());

    } catch (err) {
      if (!mountedRef.current) return;
      console.error('❌ Error fetching detection stats:', err);
      setError(err.message || 'Failed to load detection statistics');
    } finally {
      if (mountedRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [currentLotId, lastSessions, isDetectionActive]);

  // FIXED: Handle detection completion signal - ONLY refresh once on successful detection
  useEffect(() => {
    if (detectionCompleted && detectionCompleted !== detectionCompletedRef.current) {
      console.log('🎯 Detection completed - triggering single refresh');
      detectionCompletedRef.current = detectionCompleted;
      
      // Small delay to ensure backend has processed the detection
      setTimeout(() => {
        fetchFreshData(false, true); // bypassCache = true for fresh data
      }, 1500);
    }
  }, [detectionCompleted, fetchFreshData]);

  // Manual refresh handler
  const handleManualRefresh = useCallback(() => {
    console.log('🔄 Manual refresh triggered');
    fetchFreshData(true, true);
  }, [fetchFreshData]);

  // Initial load when panel opens (ONLY if we don't have data)
  useEffect(() => {
    if (panelOpen && currentLotId && !lastSessions.length) {
      console.log('🚀 Initial data load triggered');
      fetchFreshData(false, false);
    }
  }, [panelOpen, currentLotId, fetchFreshData, lastSessions]);

  // Track if detection has ever been active
  useEffect(() => {
    if (isDetectionActive) {
      setHasEverDetected(true);
    }
  }, [isDetectionActive]);

  // Clear data when lot changes
  useEffect(() => {
    setLastSessions([]);
    setError(null);
    setLastUpdateTime(null);
  }, [currentLotId]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Get current session (latest session)
  const currentSession = lastSessions.length > 0 ? lastSessions[0] : null;

  // Transform current session data for pie chart (correct vs incorrect pieces)
  const getSessionMatchingChartData = () => {
    if (!currentSession) return [];
    
    const correctPieces = currentSession.correct_pieces || 0;
    const misplacedPieces = currentSession.misplaced_pieces || 0;
    
    console.log('📊 Session pie chart data debug:', {
      correctPieces,
      misplacedPieces,
      currentSession
    });
    
    return [
      {
        id: 'correct',
        value: correctPieces,
        label: 'Correct Pieces',
        color: SESSION_MATCHING_COLORS[0] // Green
      },
      {
        id: 'misplaced',
        value: misplacedPieces,
        label: 'Misplaced Pieces',
        color: SESSION_MATCHING_COLORS[1] // Red
      }
    ].filter(item => item.value > 0); // Only show non-zero values
  };

  // Calculate session matching percentage
  const getSessionMatchingPercentage = () => {
    if (!currentSession) return 0;
    
    const correctPieces = currentSession.correct_pieces || 0;
    const misplacedPieces = currentSession.misplaced_pieces || 0;
    const totalPieces = correctPieces + misplacedPieces;
    
    return totalPieces > 0 ? Math.round((correctPieces / totalPieces) * 100) : 0;
  };

  const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const hasData = lastSessions.length > 0;
  const shouldShowStats = (isDetectionActive || hasEverDetected) && hasData && currentLotId;
  const shouldShowNoDetectionState = !isDetectionActive && !hasEverDetected;
  const shouldShowWaitingState = (!isDetectionActive && hasEverDetected) || (!currentLotId && isDetectionActive);
  const isRefreshingData = loading || refreshing;

  return (
    <>
      {/* Toggle Button */}
      <StatsToggleButton 
        isOpen={panelOpen} 
        onClick={togglePanel}
        hasData={!!hasData}
        isLoading={isRefreshingData}
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
                Current Session Statistics
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
            action={
              <Stack direction="row" spacing={1}>
                {currentLotId && (
                  <Tooltip title={refreshing ? "Refreshing..." : "Refresh data manually"}>
                    <IconButton 
                      size="small" 
                      onClick={handleManualRefresh} 
                      disabled={isRefreshingData}
                    >
                      {isRefreshingData ? 
                        <CircularProgress size={16} /> : 
                        <Refresh fontSize="small" />
                      }
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
                    ? 'Select a lot and start detection to see session statistics.'
                    : 'Session statistics are paused. Start detection again or refresh manually to see latest data.'
                  }
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Use the refresh button above to manually update session statistics when needed.
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
            {error && (isDetectionActive || hasEverDetected) && (
              <Alert severity="error" sx={{ mb: 2 }} action={
                <Button size="small" onClick={handleManualRefresh}>Retry</Button>
              }>
                {error}
              </Alert>
            )}

            {/* Loading State */}
            {loading && currentLotId && !hasData && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <Stack alignItems="center" spacing={2}>
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary">
                    Loading session statistics...
                  </Typography>
                </Stack>
              </Box>
            )}

            {/* No Data State (when lot is selected but no data) */}
            {!isRefreshingData && !hasData && !error && currentLotId && (
              <Alert severity="info" action={
                <Button size="small" onClick={handleManualRefresh}>Load Data</Button>
              }>
                No session statistics available yet for this lot. Click "Load Data" or perform detection to generate data.
              </Alert>
            )}

            {/* Stats Content - Show for detection active OR if we have session data */}
            {shouldShowStats && (
              <Stack spacing={3}>
                {/* Session Matching Pie Chart */}
                {currentSession && (currentSession.correct_pieces > 0 || currentSession.misplaced_pieces > 0) && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                      Session Detection Results
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Session #{currentSession.last_session_id} - Total Detected: {currentSession.total_detected || 0}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'center', position: 'relative' }}>
                      <PieChart
                        colors={SESSION_MATCHING_COLORS}
                        margin={{ left: 80, right: 80, top: 80, bottom: 80 }}
                        series={[
                          {
                            data: getSessionMatchingChartData(),
                            innerRadius: 58,
                            outerRadius: 75,
                            paddingAngle: 2,
                            highlightScope: { faded: 'global', highlighted: 'item' }
                          }
                        ]}
                        height={150}
                        width={150}
                        slotProps={{
                          legend: { hidden: true }
                        }}
                      >
                        <PieCenterLabel 
                          primaryText={`${getSessionMatchingPercentage()}%`}
                          secondaryText="Accuracy"
                        />
                      </PieChart>
                    </Box>


                  </Box>
                )}

                {/* Show message if no pieces detected yet */}
                {currentSession && (currentSession.correct_pieces || 0) === 0 && (currentSession.misplaced_pieces || 0) === 0 && (
                  <Alert severity="info">
                    <Typography variant="body2">
                      No pieces detected yet in this session. The pie chart will appear once pieces are detected.
                    </Typography>
                  </Alert>
                )}

                <Divider />

                {/* Session Details */}
                {currentSession && (
                  <Box>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                      <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                        Session Details
                      </Typography>
                      {refreshing && (
                        <Chip 
                          icon={<CircularProgress size={14} />}
                          label="Updating..."
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                    </Stack>
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
                              Detection Rate
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {Math.round(currentSession.detection_rate || 0)}%
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
                              Total Detected
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {currentSession.total_detected || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Session Started
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