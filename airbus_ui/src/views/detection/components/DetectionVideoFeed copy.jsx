// AdaptiveDetectionVideoFeed.jsx - Fixed to receive performance mode as props
import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  Box, 
  Alert, 
  CircularProgress, 
  Typography, 
  Chip, 
  Button,
  Stack,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip
} from "@mui/material";
import { 
  Speed as SpeedIcon, 
  Settings as SettingsIcon,
  CameraAlt as CameraIcon,
  FlashOn as FlashOnIcon,
  Assessment as AssessmentIcon 
} from "@mui/icons-material";

import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";

// Import both services
import { detectionService } from "../detectionService"; // High-performance service
import { basicDetectionService } from "../basicDetectionService";
import { systemPerformanceService } from "../systemPerformanceService";

const DetectionVideoFeed = ({
  isDetectionActive,
  onStartDetection,
  onStopDetection,
  cameraId,
  targetLabel,
  isModelLoaded: externalModelLoaded,
  onModelLoadedChange = () => {},
  detectionOptions = {},
  // NEW: Receive performance mode and system profile as props
  performanceMode: propPerformanceMode,
  systemProfile: propSystemProfile
}) => {
  // REMOVED: Performance mode state - now comes from props
  // const [performanceMode, setPerformanceMode] = useState(null);
  // const [systemProfile, setSystemProfile] = useState(null);
  // const [performanceModeLoading, setPerformanceModeLoading] = useState(true);
  // const [performanceModeError, setPerformanceModeError] = useState(null);
  
  // Use props for performance mode
  const performanceMode = propPerformanceMode;
  const systemProfile = propSystemProfile;
  const performanceModeLoading = !performanceMode; // Loading if no mode provided
  const [performanceModeError, setPerformanceModeError] = useState(null);
  
  // Initialization state
  const [isInitialized, setIsInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [initializationError, setInitializationError] = useState(null);
  
  // Model state
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  
  // Stream state for both modes
  const [videoUrl, setVideoUrl] = useState("");
  const [showControls, setShowControls] = useState(false);
  const [streamStatus, setStreamStatus] = useState({
    isLoading: false,
    error: null,
    isConnected: false
  });
  
  // Detection stats (unified for both modes)
  const [detectionStats, setDetectionStats] = useState({
    objectDetected: false,
    detectionCount: 0,
    nonTargetCount: 0,
    lastDetectionTime: null,
    avgProcessingTime: 0,
    streamQuality: 85,
    detectionFps: 5.0,
    queueDepth: 0,
    isStreamActive: false
  });
  
  // Basic mode specific state
  const [lastDetectionResult, setLastDetectionResult] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [showPerformanceDialog, setShowPerformanceDialog] = useState(false);
  const [frozenFrame, setFrozenFrame] = useState(null);
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);
  const canvasRef = useRef(null);

  // REMOVED: Performance mode determination - now comes from props
  // useEffect(() => { ... }, []);

  // Initialize the appropriate detection service based on performance mode
  useEffect(() => {
    const initializeDetectionService = async () => {
      if (!performanceMode || isInitialized || isInitializing) return;

      setIsInitializing(true);
      setInitializationError(null);

      try {
        console.log(`🚀 Initializing ${performanceMode} performance detection service...`);

        if (performanceMode === 'high') {
          // Initialize high-performance detection service
          const result = await detectionService.ensureInitialized();
          if (result.success) {
            const modelResult = await detectionService.loadModel();
            setIsModelLoaded(modelResult.success);
            onModelLoadedChange(modelResult.success);
          } else {
            throw new Error(result.message);
          }
        } else {
          // Initialize basic detection service
          const result = await basicDetectionService.initialize();
          if (result.success) {
            setIsModelLoaded(true);
            onModelLoadedChange(true);
          } else {
            throw new Error(result.message);
          }
        }

        setIsInitialized(true);
        console.log(`✅ ${performanceMode} performance detection service initialized`);

      } catch (error) {
        console.error(`❌ Error initializing ${performanceMode} detection service:`, error);
        setInitializationError(error.message);
        setIsModelLoaded(false);
        onModelLoadedChange(false);
      } finally {
        setIsInitializing(false);
      }
    };

    initializeDetectionService();
  }, [performanceMode, isInitialized, isInitializing, onModelLoadedChange]);

  // Handle detection start based on mode
  const handleStartDetection = useCallback(async () => {
    if (!cameraId || cameraId === '') {
      alert("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      alert("Please enter a target label for detection.");
      return;
    }
    
    if (!isModelLoaded) {
      alert("Detection model is not loaded. Please wait for initialization to complete.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });

    try {
      if (performanceMode === 'high') {
        // Start high-performance real-time detection
        const streamUrl = await detectionService.startOptimizedDetectionFeed(
          parseInt(cameraId), 
          targetLabel, 
          {
            detectionFps: detectionOptions.detectionFps || 5.0,
            streamQuality: detectionOptions.streamQuality || 85,
            priority: detectionOptions.priority || 1
          }
        );

        setVideoUrl(streamUrl);
        
        // Add stats listener for high-performance mode
        detectionService.addStatsListener(parseInt(cameraId), (stats) => {
          if (mountedRef.current) {
            setDetectionStats(prevStats => ({
              ...prevStats,
              ...stats,
              detectionCount: stats.objectDetected && !prevStats.objectDetected 
                ? prevStats.detectionCount + 1 
                : (stats.detectionCount || prevStats.detectionCount)
            }));
          }
        });

      } else {
        // For basic mode, start a video stream without real-time detection
        const streamUrl = `/api/video_streaming/video/basic/stream/${cameraId}?stream_quality=${detectionOptions.streamQuality || 85}`;
        setVideoUrl(streamUrl);
        
        // Initialize detection stats for basic mode
        setDetectionStats({
          objectDetected: false,
          detectionCount: 0,
          nonTargetCount: 0,
          lastDetectionTime: null,
          avgProcessingTime: 0,
          streamQuality: detectionOptions.streamQuality || 85,
          detectionFps: 0, // No continuous detection in basic mode
          queueDepth: 0,
          isStreamActive: true
        });
        
        // Clear any frozen frame from previous session
        setFrozenFrame(null);
        setLastDetectionResult(null);
      }

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      onStartDetection();

      console.log(`✅ Started ${performanceMode} detection for camera ${cameraId}`);

    } catch (error) {
      console.error(`❌ Error starting ${performanceMode} detection:`, error);
      
      if (mountedRef.current) {
        setStreamStatus({ 
          isLoading: false, 
          error: error.message || "Failed to start detection", 
          isConnected: false 
        });
        alert(`Failed to start detection: ${error.message}`);
      }
    }
  }, [cameraId, targetLabel, isModelLoaded, performanceMode, detectionOptions, onStartDetection]);

  // Handle detection stop based on mode
  const handleStopDetection = useCallback(async () => {
    if (!mountedRef.current) return;
    
    setStreamStatus(prev => ({ ...prev, isLoading: true }));

    try {
      if (performanceMode === 'high') {
        detectionService.removeStatsListener(parseInt(cameraId), () => {});
        await detectionService.stopOptimizedDetectionFeed(parseInt(cameraId), false);
      } else {
        // For basic mode, just stop the video stream
        console.log(`⏹️ Stopping basic mode stream for camera ${cameraId}`);
      }
      
      setVideoUrl("");
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: 85,
        detectionFps: 0,
        queueDepth: 0,
        isStreamActive: false
      });
      setLastDetectionResult(null);
      setFrozenFrame(null);
      
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      onStopDetection();

      console.log(`✅ Stopped ${performanceMode} detection for camera ${cameraId}`);

    } catch (error) {
      console.error(`❌ Error stopping ${performanceMode} detection:`, error);
      
      if (mountedRef.current) {
        setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop detection" }));
      }
    }
  }, [cameraId, performanceMode, onStopDetection]);

  // Handle manual detection trigger for basic mode
  const handleTriggerDetection = useCallback(async () => {
    if (performanceMode !== 'basic' || !cameraId || !targetLabel || !isDetectionActive) return;

    setIsDetecting(true);

    try {
      console.log(`🎯 Triggering manual detection for camera ${cameraId}`);
      
      // Capture current frame from video before detection
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
      }
      
      const result = await basicDetectionService.detectSingleFrame(
        parseInt(cameraId), 
        targetLabel, 
        { quality: detectionOptions.streamQuality || 85 }
      );

      if (result.success && mountedRef.current) {
        setLastDetectionResult(result.data);
        
        // Update detection stats
        setDetectionStats(prev => ({
          ...prev,
          objectDetected: result.data.detected_target,
          detectionCount: result.data.detected_target ? prev.detectionCount + 1 : prev.detectionCount,
          nonTargetCount: result.data.non_target_count || prev.nonTargetCount,
          lastDetectionTime: result.data.timestamp,
          avgProcessingTime: result.data.processing_time_ms
        }));
        
        // If we have frame data from the server, use it as frozen frame
        if (result.data.frame_data) {
          setFrozenFrame(`data:image/jpeg;base64,${result.data.frame_data}`);
        }
        
        console.log(`✅ Manual detection completed:`, {
          detected: result.data.detected_target,
          confidence: result.data.confidence,
          processingTime: result.data.processing_time_ms
        });
      }

    } catch (error) {
      console.error("❌ Error in manual detection:", error);
      alert(`Detection failed: ${error.message}`);
    } finally {
      setIsDetecting(false);
    }
  }, [performanceMode, cameraId, targetLabel, isDetectionActive, detectionOptions.streamQuality]);

  // Clear frozen frame to resume live stream
  const handleResumeLiveStream = useCallback(() => {
    setFrozenFrame(null);
    setLastDetectionResult(null);
  }, []);

  // Retry initialization
  const retryInitialization = useCallback(async () => {
    setIsInitialized(false);
    setIsInitializing(false);
    setInitializationError(null);
    setIsModelLoaded(false);
    
    // Re-trigger initialization
    // This will be handled by the useEffect above
  }, []);

  // MODIFIED: Refresh performance mode now calls parent's refresh function
  const refreshPerformanceMode = useCallback(async () => {
    try {
      // Call the parent's refresh function if available
      if (window.refreshSystemPerformance) {
        await window.refreshSystemPerformance();
      } else {
        // Fallback: direct service call
        await systemPerformanceService.getSystemProfile(true);
      }
      
      // Reset initialization to use new mode
      setIsInitialized(false);
      setIsModelLoaded(false);
    } catch (error) {
      setPerformanceModeError(error.message);
    }
  }, []);

  // Performance mode info
  const performanceModeInfo = performanceMode ? systemPerformanceService.getPerformanceModeInfo() : null;

  // MODIFIED: Show loading only if performanceMode is not provided
  if (performanceModeLoading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 3 }}>
        <CircularProgress size={40} />
        <Typography variant="body2" color="textSecondary">
          Waiting for performance mode from parent component...
        </Typography>
      </Box>
    );
  }

  // Create Basic Detection View Component
  const BasicDetectionOverlay = () => (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: frozenFrame ? 'transparent' : 'rgba(0, 0, 0, 0.7)',
        zIndex: 2
      }}
    >
      {frozenFrame && (
        <img
          src={frozenFrame}
          alt="Detection Result"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain'
          }}
        />
      )}
      
      {performanceMode === 'basic' && isDetectionActive && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            gap: 2,
            alignItems: 'center'
          }}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={handleTriggerDetection}
            disabled={isDetecting || !targetLabel}
            startIcon={isDetecting ? <CircularProgress size={20} /> : <CameraIcon />}
            sx={{ 
              backgroundColor: 'rgba(25, 118, 210, 0.9)',
              '&:hover': { backgroundColor: 'rgba(25, 118, 210, 1)' }
            }}
          >
            {isDetecting ? 'Detecting...' : 'Detect Now'}
          </Button>
          
          {frozenFrame && (
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleResumeLiveStream}
              startIcon={<FlashOnIcon />}
              sx={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                '&:hover': { backgroundColor: 'rgba(255, 255, 255, 1)' }
              }}
            >
              Resume Live
            </Button>
          )}
        </Box>
      )}
      
      {lastDetectionResult && (
        <Box
          sx={{
            position: 'absolute',
            top: 20,
            right: 20,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            p: 2,
            borderRadius: 1,
            minWidth: 200
          }}
        >
          <Typography variant="h6" gutterBottom>
            Detection Result
          </Typography>
          <Typography variant="body2">
            Target Detected: {lastDetectionResult.detected_target ? 'Yes' : 'No'}
          </Typography>
          {lastDetectionResult.confidence && (
            <Typography variant="body2">
              Confidence: {(lastDetectionResult.confidence * 100).toFixed(1)}%
            </Typography>
          )}
          <Typography variant="body2">
            Processing Time: {lastDetectionResult.processing_time_ms}ms
          </Typography>
          <Typography variant="body2">
            Objects Found: {lastDetectionResult.objects_found || 0}
          </Typography>
        </Box>
      )}
    </Box>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* Performance Mode Info */}
      <Alert 
        severity={performanceModeError ? "warning" : "info"} 
        sx={{ mb: 2, width: '100%' }}
        action={
          <Stack direction="row" spacing={1}>
            <Tooltip title="View performance details">
              <IconButton 
                size="small" 
                onClick={() => setShowPerformanceDialog(true)}
                color="inherit"
              >
                <AssessmentIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh performance analysis">
              <IconButton 
                size="small" 
                onClick={refreshPerformanceMode}
                color="inherit"
              >
                <SpeedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        }
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="body2">
            {performanceModeInfo?.icon} <strong>{performanceModeInfo?.displayName}</strong> - {performanceModeInfo?.description}
          </Typography>
        </Stack>
        {performanceModeError && (
          <Typography variant="caption" color="error" display="block">
            Warning: {performanceModeError}
          </Typography>
        )}
      </Alert>

      {/* Initialization Status */}
      {isInitializing && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing {performanceModeInfo?.displayName} detection system...
        </Alert>
      )}

      {/* Initialization Error */}
      {initializationError && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, width: '100%' }}
          action={
            <Button 
              size="small" 
              onClick={retryInitialization}
              disabled={isInitializing}
            >
              Retry
            </Button>
          }
        >
          Initialization failed: {initializationError}
        </Alert>
      )}

      {/* Stream Status Alerts */}
      {streamStatus.error && (
        <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
          {streamStatus.error}
        </Alert>
      )}

      {streamStatus.isLoading && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {isDetectionActive ? "Connecting to detection stream..." : "Stopping stream..."}
        </Alert>
      )}

      {/* Video Feed */}
      <VideoCard
        cameraActive={isDetectionActive}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
        sx={{ position: 'relative' }}
      >
        {isDetectionActive ? (
          <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
            {performanceMode === 'high' ? (
              // High-performance mode: Use LiveDetectionView as is
              <LiveDetectionView
                videoUrl={videoUrl}
                videoRef={videoRef}
                showControls={showControls}
                onStopDetection={handleStopDetection}
                detectionStats={detectionStats}
                targetLabel={targetLabel}
                streamStatus={streamStatus}
              />
            ) : (
              // Basic mode: Custom video display with manual detection
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  autoPlay
                  muted
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: frozenFrame ? 'none' : 'block'
                  }}
                  onError={(e) => {
                    console.error('Video error:', e);
                    setStreamStatus(prev => ({ 
                      ...prev, 
                      error: 'Video stream error' 
                    }));
                  }}
                />
                <BasicDetectionOverlay />
                
                {/* Hidden canvas for frame capture */}
                <canvas
                  ref={canvasRef}
                  style={{ display: 'none' }}
                />
              </>
            )}
          </Box>
        ) : (
          <CameraPlaceholder 
            onStartCamera={handleStartDetection}
            cameraId={cameraId}
            buttonText={
              isInitializing 
                ? "Initializing..." 
                : initializationError 
                  ? "Retry Initialization" 
                  : `Start ${performanceModeInfo?.displayName || 'Detection'}`
            }
            icon="detection"
            disabled={
              isInitializing ||
              !targetLabel || 
              !cameraId || 
              streamStatus.isLoading ||
              (!isModelLoaded && !initializationError)
            }
            isLoading={isInitializing || streamStatus.isLoading}
          />
        )}
      </VideoCard>

      {/* Mode Status Info */}
      {isInitialized && (
        <Box sx={{ mt: 1, width: '100%', display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip 
            label={`${performanceModeInfo?.displayName} Mode`}
            size="small" 
            color="success"
            icon={<SpeedIcon />}
          />
          <Chip 
            label={`Model: ${isModelLoaded ? 'Ready' : 'Not Loaded'}`} 
            size="small" 
            color={isModelLoaded ? 'success' : 'warning'}
          />
          {performanceMode === 'basic' && isDetectionActive && (
            <Chip 
              label="Click 'Detect Now' to analyze frame"
              size="small" 
              color="info"
              icon={<CameraIcon />}
            />
          )}
        </Box>
      )}

      {/* Performance Details Dialog */}
      <Dialog 
        open={showPerformanceDialog} 
        onClose={() => setShowPerformanceDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          System Performance Analysis
        </DialogTitle>
        <DialogContent>
          {systemProfile && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="h6" gutterBottom>
                Current Mode: {performanceModeInfo?.displayName}
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <SpeedIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="CPU Cores"
                    secondary={`${systemProfile.cpu_cores} cores`}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <AssessmentIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Memory"
                    secondary={`${systemProfile.total_memory_gb} GB`}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="GPU Available"
                    secondary={systemProfile.gpu_available ? 'Yes' : 'No'}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CameraIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Performance Score"
                    secondary={`${systemProfile.performance_score}/100`}
                  />
                </ListItem>
              </List>

              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                Mode Features:
              </Typography>
              <List dense>
                {performanceModeInfo?.features.map((feature, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={feature}
                      sx={{ pl: 2 }}
                    />
                  </ListItem>
                ))}
              </List>

              {systemPerformanceService.getPerformanceRecommendations().length > 0 && (
                <>
                  <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                    Recommendations:
                  </Typography>
                  <List dense>
                    {systemPerformanceService.getPerformanceRecommendations().map((rec, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={rec}
                          sx={{ pl: 2 }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={refreshPerformanceMode}>
            Refresh Analysis
          </Button>
          <Button onClick={() => setShowPerformanceDialog(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default DetectionVideoFeed;