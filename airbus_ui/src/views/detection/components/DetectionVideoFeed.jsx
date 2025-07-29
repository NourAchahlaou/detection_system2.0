// DetectionVideoFeed.jsx - Updated with 4-state system
import React, { useState, useEffect, useRef, useCallback } from "react";
import { Box, Alert, CircularProgress, Typography, Chip, Button } from "@mui/material";
import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";
import { detectionService } from "../detectionService";

// Detection states from service
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

const DetectionVideoFeed = ({
  isDetectionActive,
  onStartDetection,
  onStopDetection,
  cameraId,
  targetLabel,
  detectionOptions = {}
}) => {
  // State management
  const [videoUrl, setVideoUrl] = useState("");
  const [showControls, setShowControls] = useState(false);
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
  const [streamStatus, setStreamStatus] = useState({
    isLoading: false,
    error: null,
    isConnected: false
  });
  
  // Detection service state tracking
  const [detectionState, setDetectionState] = useState(detectionService.getState());
  const [isModelLoaded, setIsModelLoaded] = useState(detectionService.isModelLoaded);
  const [componentError, setComponentError] = useState(null);
  const [initializationAttempts, setInitializationAttempts] = useState(0);
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);
  const stateChangeUnsubscribe = useRef(null);

  // Subscribe to detection service state changes
  useEffect(() => {
    mountedRef.current = true;
    
    const unsubscribe = detectionService.addStateChangeListener((newState, oldState) => {
      if (!mountedRef.current) return;
      
      console.log(`🔄 VideoFeed: Detection state changed: ${oldState} → ${newState}`);
      setDetectionState(newState);
      
      // Handle state-specific logic
      switch (newState) {
        case DetectionStates.INITIALIZING:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError(null);
          break;
          
        case DetectionStates.READY:
          setIsModelLoaded(detectionService.isModelLoaded);
          if (oldState === DetectionStates.SHUTTING_DOWN) {
            // Clean up after shutdown
            setStreamStatus({ isLoading: false, error: null, isConnected: false });
            setVideoUrl("");
            setComponentError(null);
            resetDetectionStats();
          } else if (oldState === DetectionStates.INITIALIZING) {
            // Initialization completed
            setStreamStatus(prev => ({ ...prev, isLoading: false }));
            setComponentError(null);
          }
          break;
          
        case DetectionStates.RUNNING:
          setStreamStatus(prev => ({ ...prev, isLoading: false, isConnected: true }));
          setComponentError(null);
          break;
          
        case DetectionStates.SHUTTING_DOWN:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError("System is shutting down...");
          break;
      }
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
    
    return () => {
      mountedRef.current = false;
      if (stateChangeUnsubscribe.current) {
        stateChangeUnsubscribe.current();
      }
    };
  }, []);

  // Initialize detection system on component mount
  useEffect(() => {
    const initializeIfNeeded = async () => {
      // Only initialize if we're in INITIALIZING state and haven't tried yet
      if (detectionState === DetectionStates.INITIALIZING && initializationAttempts === 0) {
        console.log("🚀 VideoFeed: Starting detection system initialization...");
        setInitializationAttempts(1);
        
        try {
          await detectionService.ensureInitialized();
        } catch (error) {
          console.error("❌ VideoFeed: Initialization failed:", error);
          if (mountedRef.current) {
            setComponentError(`Initialization failed: ${error.message}`);
          }
        }
      }
    };

    initializeIfNeeded();
  }, [detectionState, initializationAttempts]);

  // Reset detection stats helper
  const resetDetectionStats = useCallback(() => {
    setDetectionStats({
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
  }, []);

  // Manual retry initialization
  const handleRetryInitialization = useCallback(async () => {
    console.log("🔄 VideoFeed: Manual initialization retry requested");
    setComponentError(null);
    setInitializationAttempts(prev => prev + 1);
    
    try {
      // Reset service to initializing state
      detectionService.resetToInitializing('Manual retry from VideoFeed');
      
      // Wait a moment for state to settle
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Try initialization
      await detectionService.ensureInitialized();
    } catch (error) {
      console.error("❌ VideoFeed: Retry initialization failed:", error);
      if (mountedRef.current) {
        setComponentError(`Retry failed: ${error.message}`);
      }
    }
  }, []);

  // Stats listener callback
  const handleStatsUpdate = useCallback((newStats) => {
    if (!mountedRef.current || detectionState === DetectionStates.SHUTTING_DOWN) return;
    
    setDetectionStats(prevStats => ({
      ...prevStats,
      ...newStats,
      detectionCount: newStats.objectDetected && !prevStats.objectDetected 
        ? prevStats.detectionCount + 1 
        : (newStats.detectionCount || prevStats.detectionCount)
    }));
  }, [detectionState]);

  // Enhanced detection start with proper state validation
  const handleStartDetection = async () => {
    if (!cameraId || cameraId === '') {
      setComponentError("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      setComponentError("Please enter a target label first.");
      return;
    }

    // Check current detection service state
    const currentState = detectionService.getState();
    console.log(`🎯 VideoFeed: Attempting to start detection. Current state: ${currentState}`);

    // Validate state before starting
    if (currentState === DetectionStates.INITIALIZING) {
      setComponentError("System is still initializing. Please wait...");
      return;
    }

    if (currentState === DetectionStates.SHUTTING_DOWN) {
      setComponentError("System is shutting down. Please wait for it to complete.");
      return;
    }

    if (!detectionService.canStart()) {
      setComponentError(`Cannot start detection. Current state: ${currentState}. System must be READY.`);
      return;
    }

    if (!detectionService.isModelLoaded) {
      setComponentError("Detection model is not loaded. Please wait or try refreshing.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });
    setComponentError(null);

    try {
      const streamUrl = await detectionService.startOptimizedDetectionFeed(
        parseInt(cameraId), 
        targetLabel, 
        {
          detectionFps: detectionOptions.detectionFps || 5.0,
          streamQuality: detectionOptions.streamQuality || 85,
          priority: detectionOptions.priority || 1
        }
      );

      if (!mountedRef.current) return;

      setVideoUrl(streamUrl);
      detectionService.addStatsListener(parseInt(cameraId), handleStatsUpdate);
      
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: detectionOptions.streamQuality || 85,
        detectionFps: detectionOptions.detectionFps || 5.0,
        queueDepth: 0,
        isStreamActive: true
      });

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      onStartDetection();

      console.log(`✅ VideoFeed: Started optimized detection for camera ${cameraId} with target: ${targetLabel}`);

    } catch (error) {
      console.error("❌ VideoFeed: Error starting optimized detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus({ 
        isLoading: false, 
        error: error.message || "Failed to start detection", 
        isConnected: false 
      });
      setComponentError(`Failed to start detection: ${error.message}`);
    }
  };

  // Handle stopping detection
  const handleStopDetection = async () => {
    if (!mountedRef.current) return;
    
    console.log(`🛑 VideoFeed: Stopping detection for camera ${cameraId}`);
    setStreamStatus(prev => ({ ...prev, isLoading: true }));

    try {
      detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      await detectionService.stopOptimizedDetectionFeed(parseInt(cameraId), false); // Don't shutdown system
      
      if (!mountedRef.current) return;
      
      setVideoUrl("");
      resetDetectionStats();
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      setComponentError(null);
      onStopDetection();

      console.log(`✅ VideoFeed: Stopped optimized detection for camera ${cameraId}`);

    } catch (error) {
      console.error("❌ VideoFeed: Error stopping optimized detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop detection" }));
      setComponentError(`Failed to stop detection: ${error.message}`);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraId) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
    };
  }, [cameraId, handleStatsUpdate]);

  // Get appropriate button text based on state
  const getButtonText = () => {
    switch (detectionState) {
      case DetectionStates.INITIALIZING:
        return "Initializing System...";
      case DetectionStates.READY:
        return componentError ? "Retry" : "Start Detection";
      case DetectionStates.RUNNING:
        return "System Running";
      case DetectionStates.SHUTTING_DOWN:
        return "Shutting Down...";
      default:
        return "Start Detection";
    }
  };

  // Determine if button should be disabled
  const isButtonDisabled = () => {
    return (
      detectionState === DetectionStates.INITIALIZING ||
      detectionState === DetectionStates.SHUTTING_DOWN ||
      (detectionState === DetectionStates.RUNNING && isDetectionActive) ||
      !targetLabel || 
      !cameraId || 
      streamStatus.isLoading ||
      (!isModelLoaded && detectionState !== DetectionStates.INITIALIZING)
    );
  };

  // Get appropriate loading state
  const isLoading = () => {
    return (
      detectionState === DetectionStates.INITIALIZING ||
      detectionState === DetectionStates.SHUTTING_DOWN ||
      streamStatus.isLoading
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* State-based Status Alerts */}
      {detectionState === DetectionStates.INITIALIZING && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing detection system... Please wait.
        </Alert>
      )}

      {detectionState === DetectionStates.SHUTTING_DOWN && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          System is shutting down... This may take a moment.
        </Alert>
      )}

      {componentError && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, width: '100%' }}
          action={
            detectionState === DetectionStates.INITIALIZING ? (
              <Button 
                color="inherit" 
                size="small" 
                onClick={handleRetryInitialization}
                disabled={initializationAttempts >= 3}
              >
                {initializationAttempts >= 3 ? 'Max Retries' : 'Retry'}
              </Button>
            ) : null
          }
        >
          {componentError}
          {initializationAttempts >= 3 && (
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Maximum retry attempts reached. Please refresh the page.
            </Typography>
          )}
        </Alert>
      )}

      {/* Stream Status Alerts */}
      {streamStatus.error && (
        <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
          {streamStatus.error}
        </Alert>
      )}

      {streamStatus.isLoading && detectionState !== DetectionStates.SHUTTING_DOWN && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {isDetectionActive ? "Connecting to optimized detection stream..." : "Stopping stream..."}
        </Alert>
      )}

      <VideoCard
        cameraActive={isDetectionActive && detectionState === DetectionStates.RUNNING}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isDetectionActive && detectionState === DetectionStates.RUNNING ? (
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
          <CameraPlaceholder 
            onStartCamera={componentError && detectionState === DetectionStates.INITIALIZING 
              ? handleRetryInitialization 
              : handleStartDetection}
            cameraId={cameraId}
            buttonText={getButtonText()}
            icon="detection"
            disabled={isButtonDisabled()}
            isLoading={isLoading()}
          />
        )}
      </VideoCard>

      {/* System Status Info */}
      <Box sx={{ mt: 1, width: '100%', display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip 
          label={`State: ${detectionState}`} 
          size="small" 
          color={
            detectionState === DetectionStates.READY ? 'success' :
            detectionState === DetectionStates.RUNNING ? 'primary' :
            detectionState === DetectionStates.INITIALIZING ? 'warning' :
            'error'
          }
        />
        
        <Chip 
          label={`Model: ${isModelLoaded ? 'Loaded' : 'Not Loaded'}`} 
          size="small" 
          color={isModelLoaded ? 'success' : 'warning'}
        />
        
        {detectionState === DetectionStates.RUNNING && (
          <Chip 
            label={`Active Streams: ${detectionService.currentStreams.size}`} 
            size="small" 
            color="info"
          />
        )}
        
        {initializationAttempts > 0 && detectionState === DetectionStates.INITIALIZING && (
          <Chip 
            label={`Attempts: ${initializationAttempts}`} 
            size="small" 
            color="warning"
          />
        )}
      </Box>
    </div>
  );
};

export default DetectionVideoFeed;