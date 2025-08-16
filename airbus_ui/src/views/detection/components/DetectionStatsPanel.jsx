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

// Debounce utility function
const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
};

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
const StatsToggleButton = ({ isOpen, onClick, hasData, isLoading, isDetectionActive, hasActiveSession }) => {
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
          '&::after': hasActiveSession && !isLoading ? {
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
      No Detection Running
    </Typography>
    <Typography variant="body2" color="text.disabled" sx={{ mb: 3, maxWidth: 300, mx: 'auto' }}>
      Start detection to see real-time statistics about your current session results.
    </Typography>
    <Alert severity="info" sx={{ mb: 2 }}>
      <Typography variant="body2">
        <strong>Start Detection</strong> to begin capturing session statistics.
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
  isStreamFrozen = false, // New prop to know if stream is frozen
}) => {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [error, setError] = useState(null);
  const [internalOpen, setInternalOpen] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);
  
  // Refs for managing updates and caching
  const mountedRef = useRef(true);
  const detectionCompletedRef = useRef(null);
  const sessionCache = useRef(new Map()); // Add session cache
  const requestInProgress = useRef(false); // Prevent duplicate requests
  
  const isControlled = typeof isOpen === 'boolean' && typeof onToggle === 'function';
  const panelOpen = isControlled ? isOpen : internalOpen;

  const togglePanel = () => {
    if (isControlled) {
      onToggle(!isOpen);
    } else {
      setInternalOpen(prev => !prev);
    }
  };

  // Clear session data when detection stops or stream is unfrozen
  useEffect(() => {
    if (!isDetectionActive && !isStreamFrozen) {
      console.log('🧹 Clearing session data - detection stopped and stream not frozen');
      setCurrentSession(null);
      setError(null);
      setLastUpdateTime(null);
      sessionCache.current.clear();
    }
  }, [isDetectionActive, isStreamFrozen]);

  // ULTRA-OPTIMIZED: Direct lot-specific fetch using the correct endpoint
  const fetchSessionData = useCallback(async (isManualRefresh = false) => {
    if (!currentLotId) {
      console.log('🔍 Skipping stats fetch - no lot ID');
      return;
    }

    // Prevent duplicate requests
    if (requestInProgress.current) {
      console.log('🔍 Request already in progress, skipping');
      return;
    }

    // Check cache first (5 second cache for non-manual refreshes)
    const cacheKey = currentLotId.toString();
    if (!isManualRefresh && sessionCache.current.has(cacheKey)) {
      const cached = sessionCache.current.get(cacheKey);
      if (Date.now() - cached.timestamp < 5000) { // 5s cache
        console.log('📋 Using cached session data');
        setCurrentSession(cached.data);
        setLastUpdateTime(cached.updateTime);
        return;
      }
    }

    requestInProgress.current = true;

    if (isManualRefresh) {
      setRefreshing(true);
    } else if (!currentSession) {
      setLoading(true);
    }
    setError(null);
    
    try {
      console.log('🚀 ULTRA-FAST: Fetching DIRECT session data for lot:', currentLotId);

      // 🔥 CRITICAL FIX: Use the direct endpoint instead of fetching all lots
      const sessionResult = await detectionStatisticsService.getLastSessionForLot(currentLotId);

      if (!mountedRef.current) return;

      if (sessionResult.success && sessionResult.data) {
        console.log("⚡ INSTANT: Session data loaded:", sessionResult.data);
        
        const updateTime = new Date().toLocaleTimeString();
        
        // Update cache
        sessionCache.current.set(cacheKey, {
          data: sessionResult.data,
          timestamp: Date.now(),
          updateTime: updateTime
        });

        // Batch state updates to reduce re-renders
        setCurrentSession(sessionResult.data);
        setLastUpdateTime(updateTime);
        setError(null);
      } else {
        console.log("📊 No session found for current lot");
        setCurrentSession(null);
        setError(sessionResult.error || 'No session data available');
      }

    } catch (err) {
      if (!mountedRef.current) return;
      console.error('❌ Error fetching session data:', err);
      setError(err.message || 'Failed to load session statistics');
    } finally {
      if (mountedRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
      requestInProgress.current = false;
    }
  }, [currentLotId, currentSession]);

  // Add debounced version to prevent rapid-fire requests
  const debouncedFetchSessionData = useCallback(
    debounce((isManualRefresh = false) => {
      fetchSessionData(isManualRefresh);
    }, 300),
    [fetchSessionData]
  );

  // OPTIMIZATION: Immediate response to detection completion
  useEffect(() => {
    if (detectionCompleted && detectionCompleted !== detectionCompletedRef.current) {
      console.log('🎯 INSTANT: Detection completed - immediate data fetch');
      detectionCompletedRef.current = detectionCompleted;
      
      // Open panel immediately
      if (isControlled) {
        onToggle(true);
      } else {
        setInternalOpen(true);
      }
      
      // Clear cache for this lot to ensure fresh data
      if (currentLotId) {
        sessionCache.current.delete(currentLotId.toString());
      }
      
      // CRITICAL: No setTimeout, no delay - fetch immediately
      fetchSessionData(false);
    }
  }, [detectionCompleted, fetchSessionData, isControlled, onToggle, currentLotId]);

  // Manual refresh handler
  const handleManualRefresh = useCallback(() => {
    console.log('🔄 Manual refresh triggered');
    if (currentLotId) {
      sessionCache.current.delete(currentLotId.toString());
    }
    fetchSessionData(true);
  }, [fetchSessionData, currentLotId]);

  // Clear data when lot changes
  useEffect(() => {
    console.log('🔄 Lot changed, clearing session data');
    setCurrentSession(null);
    setError(null);
    setLastUpdateTime(null);
    sessionCache.current.clear();
  }, [currentLotId]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestInProgress.current = false;
    };
  }, []);

  // OPTIMIZATION: Inline calculations to avoid function overhead
  const getSessionMatchingChartData = () => {
    if (!currentSession) return [];
    
    const correctPieces = currentSession.correct_pieces || 0;
    const misplacedPieces = currentSession.misplaced_pieces || 0;
    
    const data = [];
    if (correctPieces > 0) {
      data.push({
        id: 'correct',
        value: correctPieces,
        label: 'Correct Pieces',
        color: SESSION_MATCHING_COLORS[0]
      });
    }
    if (misplacedPieces > 0) {
      data.push({
        id: 'misplaced',
        value: misplacedPieces,
        label: 'Misplaced Pieces',
        color: SESSION_MATCHING_COLORS[1]
      });
    }
    
    return data;
  };

  // Calculate session matching percentage
  const getSessionMatchingPercentage = () => {
    if (!currentSession) return 0;
    
    const correctPieces = currentSession.correct_pieces || 0;
    const misplacedPieces = currentSession.misplaced_pieces || 0;
    const totalPieces = correctPieces + misplacedPieces;
    
    return totalPieces > 0 ? Math.round((correctPieces / totalPieces) * 100) : 0;
  };

  // OPTIMIZATION: Inline date formatting to avoid function calls
  const formatDateTime = (dateString) => {
    return dateString ? new Date(dateString).toLocaleString() : 'N/A';
  };

  // Determine what to show based on current state
  const hasActiveSession = currentSession && (isDetectionActive || isStreamFrozen);
  const shouldShowStats = hasActiveSession && currentLotId;
  const shouldShowNoDetectionState = !isDetectionActive && !isStreamFrozen;
  const isRefreshingData = loading || refreshing;

  // OPTIMIZATION: Pre-calculate chart data and values
  const chartData = getSessionMatchingChartData();
  const hasChartData = chartData.length > 0;

  return (
    <>
      {/* Toggle Button */}
      <StatsToggleButton 
        isOpen={panelOpen} 
        onClick={togglePanel}
        hasData={!!hasActiveSession}
        isLoading={isRefreshingData}
        isDetectionActive={isDetectionActive}
        hasActiveSession={hasActiveSession}
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
                Detection Session Results
                {hasActiveSession && (
                  <Chip 
                    label={isDetectionActive ? "LIVE" : "FROZEN"} 
                    size="small" 
                    color={isDetectionActive ? "success" : "warning"} 
                    sx={{ ml: 1, fontSize: '0.75rem' }}
                  />
                )}
              </Typography>
            }
            action={
              <Stack direction="row" spacing={1}>
                {currentLotId && hasActiveSession && (
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

            {/* Error State */}
            {error && hasActiveSession && (
              <Alert severity="error" sx={{ mb: 2 }} action={
                <Button size="small" onClick={handleManualRefresh}>Retry</Button>
              }>
                {error}
              </Alert>
            )}

            {/* Loading State */}
            {loading && currentLotId && !currentSession && hasActiveSession && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <Stack alignItems="center" spacing={2}>
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary">
                    Loading session data...
                  </Typography>
                </Stack>
              </Box>
            )}

            {/* No Session Data State */}
            {!isRefreshingData && !currentSession && !error && currentLotId && hasActiveSession && (
              <Alert severity="info" action={
                <Button size="small" onClick={handleManualRefresh}>Refresh</Button>
              }>
                No session data available yet. Detection may still be processing.
              </Alert>
            )}

            {/* Stats Content - Show only when we have active session data */}
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
                      No pieces detected yet in this session. The chart will appear once pieces are detected.
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
                              label={currentSession.is_target_match ? 'Completed' : isDetectionActive ? 'Active' : 'Frozen'}
                              color={currentSession.is_target_match ? 'success' : isDetectionActive ? 'primary' : 'warning'}
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
                              Correct Pieces
                            </Typography>
                            <Typography variant="body2" fontWeight="bold" color="success.main">
                              {currentSession.correct_pieces || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Misplaced Pieces
                            </Typography>
                            <Typography variant="body2" fontWeight="bold" color="error.main">
                              {currentSession.misplaced_pieces || 0}
                            </Typography>
                          </Grid>
                          <Grid item xs={12}>
                            <Typography variant="caption" color="textSecondary">
                              Session Started
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {formatDateTime(currentSession.last_session_time)}
                            </Typography>
                          </Grid>
                          {lastUpdateTime && (
                            <Grid item xs={12}>
                              <Typography variant="caption" color="textSecondary">
                                Last Updated
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {lastUpdateTime}
                              </Typography>
                            </Grid>
                          )}
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