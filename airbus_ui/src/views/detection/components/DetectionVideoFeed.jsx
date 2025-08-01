// DetectionVideoFeed.jsx - Enhanced with Basic Mode Logic and Freeze/Unfreeze Controls
import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  Box, 
  Alert, 
  CircularProgress, 
  Typography, 
  Button

} from "@mui/material";

import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";
import { detectionService } from "../service/DetectionService";

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
    isStreamActive: false,
    isFrozen: false,
    mode: 'basic'
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
  
  // Adaptive system state
  const [systemProfile, setSystemProfile] = useState(null);
  const [currentMode, setCurrentMode] = useState('basic');
  const [isBasicMode, setIsBasicMode] = useState(true);
  const [modeTransitioning, setModeTransitioning] = useState(false);
  
  // Basic mode specific state
  const [isStreamFrozen, setIsStreamFrozen] = useState(false);
  const [onDemandDetecting, setOnDemandDetecting] = useState(false);
  const [lastDetectionResult, setLastDetectionResult] = useState(null);
  


  
  // Health check tracking
  const [healthCheckStatus, setHealthCheckStatus] = useState({
    initial: false,
    postShutdown: false,
    lastCheck: null
  });
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);
  const stateChangeUnsubscribe = useRef(null);
  const profileUpdateUnsubscribe = useRef(null);
  const freezeListenerUnsubscribe = useRef(null);
  const initializationAttempted = useRef(false);

  // Subscribe to detection service state changes
  useEffect(() => {
    mountedRef.current = true;
    
    const unsubscribe = detectionService.addStateChangeListener((newState, oldState) => {
      if (!mountedRef.current) return;
      
      console.log(`🔄 VideoFeed: Detection state changed: ${oldState} → ${newState}`);
      setDetectionState(newState);
      
      // Update model loaded status
      setIsModelLoaded(detectionService.isModelLoaded);
      
      // Handle state-specific logic
      switch (newState) {
        case DetectionStates.INITIALIZING:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError(null);
          setModeTransitioning(true);
          // Reset health check status
          setHealthCheckStatus(prev => ({
            ...prev,
            initial: false,
            postShutdown: false
          }));
          break;
          
        case DetectionStates.READY:
          setModeTransitioning(false);
          if (oldState === DetectionStates.SHUTTING_DOWN) {
            // Clean up after shutdown
            setStreamStatus({ isLoading: false, error: null, isConnected: false });
            setVideoUrl("");
            setComponentError(null);
            resetDetectionStats();
            setIsStreamFrozen(false);
            setLastDetectionResult(null);
            
            // Mark that we need a post-shutdown health check
            setHealthCheckStatus(prev => ({
              ...prev,
              postShutdown: false
            }));
          } else if (oldState === DetectionStates.INITIALIZING) {
            // Initialization completed
            setStreamStatus(prev => ({ ...prev, isLoading: false }));
            setComponentError(null);
            
            // Mark that we need an initial health check
            setHealthCheckStatus(prev => ({
              ...prev,
              initial: false
            }));
          }
          break;
          
        case DetectionStates.RUNNING:
          setStreamStatus(prev => ({ ...prev, isLoading: false, isConnected: true }));
          setComponentError(null);
          setModeTransitioning(false);
          break;
          
        case DetectionStates.SHUTTING_DOWN:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError("System is shutting down...");
          setModeTransitioning(true);
          setIsStreamFrozen(false);
          break;
      }
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
    
    // Get initial state and model status
    setDetectionState(detectionService.getState());
    setIsModelLoaded(detectionService.isModelLoaded);
    
    return () => {
      mountedRef.current = false;
      if (stateChangeUnsubscribe.current) {
        stateChangeUnsubscribe.current();
      }
    };
  }, []);

  // Subscribe to system profile updates
  useEffect(() => {
    const unsubscribe = detectionService.addProfileUpdateListener((profileData) => {
      if (!mountedRef.current) return;
      
      console.log(`📊 VideoFeed: System profile updated`, profileData);
      setSystemProfile(profileData.profile);
      setCurrentMode(profileData.streamingType);
      setIsBasicMode(profileData.streamingType === 'basic');
      
      // Update detection stats mode
      setDetectionStats(prev => ({
        ...prev,
        mode: profileData.streamingType
      }));
    });
    
    profileUpdateUnsubscribe.current = unsubscribe;
    
    // Get initial profile
    const initialProfile = detectionService.getSystemProfile();
    const initialMode = detectionService.getCurrentStreamingType();
    if (initialProfile) {
      setSystemProfile(initialProfile);
      setCurrentMode(initialMode);
      setIsBasicMode(initialMode === 'basic');
    }
    
    return () => {
      if (profileUpdateUnsubscribe.current) {
        profileUpdateUnsubscribe.current();
      }
    };
  }, []);

  // Subscribe to freeze/unfreeze events for basic mode
  useEffect(() => {
    if (!isBasicMode || !cameraId) return;
    
    const unsubscribe = detectionService.addFreezeListener((freezeEvent) => {
      if (!mountedRef.current || freezeEvent.cameraId !== parseInt(cameraId)) return;
      
      console.log(`🧊 VideoFeed: Freeze event for camera ${cameraId}:`, freezeEvent);
      setIsStreamFrozen(freezeEvent.status === 'frozen');
      
      // Update detection stats
      setDetectionStats(prev => ({
        ...prev,
        isFrozen: freezeEvent.status === 'frozen'
      }));
    });
    
    freezeListenerUnsubscribe.current = unsubscribe;

    
    // Check initial freeze status
    if (detectionService.isStreamFrozen(cameraId)) {
      setIsStreamFrozen(true);
      setDetectionStats(prev => ({ ...prev, isFrozen: true }));
    }
    
    return () => {
      if (freezeListenerUnsubscribe.current) {
        freezeListenerUnsubscribe.current();
      }
  
    };
  }, [isBasicMode, cameraId]);

  // Initialize detection system on component mount
  useEffect(() => {
    const initializeIfNeeded = async () => {
      if (detectionState === DetectionStates.INITIALIZING && !initializationAttempted.current) {
        console.log("🚀 VideoFeed: Starting detection system initialization...");
        initializationAttempted.current = true;
        setInitializationAttempts(1);
        
        try {
          const initResult = await detectionService.ensureInitialized();
          
          if (initResult.success && mountedRef.current) {
            console.log("✅ VideoFeed: Detection system initialized successfully");
            setComponentError(null);
            
            setTimeout(() => {
              if (mountedRef.current && detectionState === DetectionStates.READY) {
                performInitialHealthCheck();
              }
            }, 1000);
          }
        } catch (error) {
          console.error("❌ VideoFeed: Initialization failed:", error);
          if (mountedRef.current) {
            setComponentError(`Initialization failed: ${error.message}`);
          }
        }
      }
    };

    initializeIfNeeded();
  }, [detectionState]);

  // Watch for READY state transitions to trigger appropriate health checks
  useEffect(() => {
    const handleReadyStateTransition = async () => {
      if (detectionState !== DetectionStates.READY) return;
      
      const serviceStatus = detectionService.getDetailedStatus();
      
      if (!healthCheckStatus.initial && !serviceStatus.hasPerformedInitialHealthCheck) {
        console.log("🩺 VideoFeed: Triggering initial health check...");
        await performInitialHealthCheck();
      }
      else if (!healthCheckStatus.postShutdown && !serviceStatus.hasPerformedPostShutdownCheck) {
        console.log("🩺 VideoFeed: Triggering post-shutdown health check...");
        await performPostShutdownHealthCheck();
      }
    };

    const timer = setTimeout(handleReadyStateTransition, 500);
    return () => clearTimeout(timer);
  }, [detectionState, healthCheckStatus.initial, healthCheckStatus.postShutdown]);

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
      isStreamActive: false,
      isFrozen: false,
      mode: currentMode
    });
    setLastDetectionResult(null);
  }, [currentMode]);

  // Initial health check function
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckStatus.initial) return;

    try {
      console.log("🩺 VideoFeed: Performing initial health check...");
      const health = await detectionService.checkOptimizedHealth(true, false);
      
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          initial: true,
          lastCheck: Date.now()
        }));
        
        console.log("✅ VideoFeed: Initial health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ VideoFeed: Initial health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          initial: true,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.initial]);

  // Post-shutdown health check function
  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckStatus.postShutdown) return;

    try {
      console.log("🩺 VideoFeed: Performing post-shutdown health check...");
      const health = await detectionService.checkOptimizedHealth(false, true);
      
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          postShutdown: true,
          lastCheck: Date.now()
        }));
        
        console.log("✅ VideoFeed: Post-shutdown health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ VideoFeed: Post-shutdown health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          postShutdown: true,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.postShutdown]);

  // Manual retry initialization
  const handleRetryInitialization = useCallback(async () => {
    console.log("🔄 VideoFeed: Manual initialization retry requested");
    setComponentError(null);
    setInitializationAttempts(prev => prev + 1);
    initializationAttempted.current = false;
    
    setHealthCheckStatus({
      initial: false,
      postShutdown: false,
      lastCheck: null
    });
    
    try {
      detectionService.resetToInitializing('Manual retry from VideoFeed');
      await new Promise(resolve => setTimeout(resolve, 500));
      const initResult = await detectionService.ensureInitialized();
      
      if (initResult.success && mountedRef.current) {
        console.log("✅ VideoFeed: Retry initialization successful");
        setComponentError(null);
      }
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
      mode: currentMode,
      detectionCount: newStats.objectDetected && !prevStats.objectDetected 
        ? prevStats.detectionCount + 1 
        : (newStats.detectionCount || prevStats.detectionCount)
    }));
  }, [detectionState, currentMode]);

  // Enhanced detection start with adaptive mode selection
  const handleStartDetection = async () => {
    if (!cameraId || cameraId === '') {
      setComponentError("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      setComponentError("Please enter a target label first.");
      return;
    }

    console.log(`🎯 VideoFeed: Attempting to start detection. Current state: ${detectionState}, Mode: ${currentMode}`);

    const currentDetectionState = detectionService.getState();
    
    if (currentDetectionState === DetectionStates.INITIALIZING) {
      setComponentError("System is still initializing. Please wait...");
      return;
    }

    if (currentDetectionState === DetectionStates.SHUTTING_DOWN) {
      setComponentError("System is shutting down. Please wait for it to complete.");
      return;
    }

    if (currentDetectionState !== DetectionStates.READY) {
      setComponentError(`Cannot start detection. Current state: ${currentDetectionState}. System must be READY.`);
      return;
    }

    if (!detectionService.isModelLoaded) {
      setComponentError("Detection model is not loaded. Please wait or try refreshing.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });
    setComponentError(null);
    setModeTransitioning(true);

    try {
      let streamUrl;
      
      // Use the auto mode detection feed which handles fallback internally
      console.log(`🎯 VideoFeed: Starting detection with auto-mode handling for camera ${cameraId}`);
      
      try {
        // Try to start with auto mode - this will handle fallback internally
        streamUrl = await detectionService.startDetectionFeedWithAutoMode(
          parseInt(cameraId), 
          targetLabel, 
          {
            detectionFps: detectionOptions.detectionFps || 5.0,
            streamQuality: detectionOptions.streamQuality || 85,
            priority: detectionOptions.priority || 1
          }
        );
      } catch (autoModeError) {
        console.log(`⚠️ Auto mode failed, explicitly trying basic mode: ${autoModeError.message}`);
        
        // If auto mode fails, explicitly try basic mode
        streamUrl = await detectionService.startBasicDetectionFeed(
          parseInt(cameraId), 
          targetLabel, 
          {
            streamQuality: detectionOptions.streamQuality || 85
          }
        );
      }

      if (!mountedRef.current) return;

      setVideoUrl(streamUrl);
      
      // Add stats listener
      detectionService.addStatsListener(parseInt(cameraId), handleStatsUpdate);
      
      // Get the current mode after starting (might have changed due to fallback)
      const actualMode = detectionService.getCurrentStreamingType();
      setCurrentMode(actualMode);
      setIsBasicMode(actualMode === 'basic');
      
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: detectionOptions.streamQuality || 85,
        detectionFps: detectionOptions.detectionFps || 5.0,
        queueDepth: 0,
        isStreamActive: true,
        isFrozen: false,
        mode: actualMode
      });

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      setModeTransitioning(false);
      onStartDetection();

      console.log(`✅ VideoFeed: Started ${actualMode} detection for camera ${cameraId}`);

    } catch (error) {
      console.error("❌ VideoFeed: Error starting detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus({ 
        isLoading: false, 
        error: error.message || "Failed to start detection", 
        isConnected: false 
      });
      setComponentError(`Failed to start detection: ${error.message}`);
      setModeTransitioning(false);
    }
  };


  // Handle stopping detection
  const handleStopDetection = async () => {
    if (!mountedRef.current) return;
    
    console.log(`🛑 VideoFeed: Stopping detection for camera ${cameraId}`);
    setStreamStatus(prev => ({ ...prev, isLoading: true }));
    setModeTransitioning(true);

    try {
      // Remove stats listener first
      detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      
      // Use the generic stop method that handles both modes
      await detectionService.stopDetectionFeed(parseInt(cameraId), false);
      
      if (!mountedRef.current) return;
      
      setVideoUrl("");
      resetDetectionStats();
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      setComponentError(null);
      setModeTransitioning(false);
      setIsStreamFrozen(false);
      setLastDetectionResult(null);
      onStopDetection();

      console.log(`✅ VideoFeed: Stopped detection for camera ${cameraId}`);

      // Schedule post-shutdown health check
      setTimeout(() => {
        if (mountedRef.current && detectionState === DetectionStates.READY) {
          console.log("🩺 VideoFeed: Scheduling post-shutdown health check...");
          setHealthCheckStatus(prev => ({
            ...prev,
            postShutdown: false
          }));
          performPostShutdownHealthCheck();
        }
      }, 2000);

    } catch (error) {
      console.error("❌ VideoFeed: Error stopping detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop detection" }));
      setComponentError(`Failed to stop detection: ${error.message}`);
      setModeTransitioning(false);
    }
  };


  // Manual mode switching functions
  const handleSwitchToBasic = useCallback(async () => {
    if (detectionState === DetectionStates.RUNNING) {
      setComponentError("Please stop detection before switching modes.");
      return;
    }

    setModeTransitioning(true);
    try {
      await detectionService.switchToBasicMode();
      console.log("✅ Switched to basic mode");
    } catch (error) {
      console.error("❌ Error switching to basic mode:", error);
      setComponentError(`Failed to switch to basic mode: ${error.message}`);
    } finally {
      setModeTransitioning(false);
    }
  }, [detectionState]);

  const handleSwitchToOptimized = useCallback(async () => {
    if (detectionState === DetectionStates.RUNNING) {
      setComponentError("Please stop detection before switching modes.");
      return;
    }

    setModeTransitioning(true);
    try {
      await detectionService.switchToOptimizedMode();
      console.log("✅ Switched to optimized mode");
    } catch (error) {
      console.error("❌ Error switching to optimized mode:", error);
      setComponentError(`Failed to switch to optimized mode: ${error.message}`);
    } finally {
      setModeTransitioning(false);
    }
  }, [detectionState]);

  const handleEnableAutoMode = useCallback(async () => {
    setModeTransitioning(true);
    try {
      await detectionService.enableAutoMode();
      console.log("✅ Auto mode enabled");
    } catch (error) {
      console.error("❌ Error enabling auto mode:", error);
      setComponentError(`Failed to enable auto mode: ${error.message}`);
    } finally {
      setModeTransitioning(false);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraId) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
    };
  }, [cameraId, handleStatsUpdate]);

  // Get appropriate button text based on state and mode
  const getButtonText = () => {
    if (modeTransitioning) return "Switching Mode...";
    
    switch (detectionState) {
      case DetectionStates.INITIALIZING:
        return "Initializing System...";
      case DetectionStates.READY:
        return componentError ? "Retry" : `Start ${isBasicMode ? 'Basic Stream' : 'Optimized Detection'}`;
      case DetectionStates.RUNNING:
        return "System Running";
      case DetectionStates.SHUTTING_DOWN:
        return "Shutting Down...";
      default:
        return "Start Stream";
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
      modeTransitioning ||
      (!isModelLoaded && detectionState !== DetectionStates.INITIALIZING)
    );
  };

  // Get appropriate loading state
  const isLoading = () => {
    return (
      detectionState === DetectionStates.INITIALIZING ||
      detectionState === DetectionStates.SHUTTING_DOWN ||
      streamStatus.isLoading ||
      modeTransitioning
    );
  };

  // Get health check status for display


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

{modeTransitioning && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Switching between detection modes... Please wait.
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
          {isDetectionActive ? `Connecting to ${currentMode} stream...` : "Stopping stream..."}
        </Alert>
      )}

      {/* Basic Mode Specific Alerts */}
      {isBasicMode && isStreamFrozen && (
        <Alert severity="info" sx={{ mb: 2, width: '100%' }}>
          <Typography variant="body2">
            Stream is frozen for detection. Use controls below to unfreeze or perform detection.
          </Typography>
        </Alert>
      )}

      {onDemandDetecting && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Performing on-demand detection...
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
            currentMode={currentMode}
            isBasicMode={isBasicMode}
            isStreamFrozen={isStreamFrozen}
            lastDetectionResult={lastDetectionResult}
            onDemandDetecting={onDemandDetecting}
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



      {/* Manual Mode Control Buttons */}
      {detectionState === DetectionStates.READY && !detectionService.autoModeEnabled && (
        <Box sx={{ mt: 1, width: '100%', display: 'flex', gap: 1, justifyContent: 'center' }}>
          <Button
            size="small"
            variant="outlined"
            onClick={handleSwitchToBasic}
            disabled={isBasicMode || modeTransitioning}
            color="warning"
          >
            Switch to Basic
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={handleSwitchToOptimized}
            disabled={!isBasicMode || modeTransitioning}
            color="success"
          >
            Switch to Optimized
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={handleEnableAutoMode}
            disabled={modeTransitioning}
            color="primary"
          >
            Enable Auto Mode
          </Button>
        </Box>
      )}


    </div>
  );
};

export default DetectionVideoFeed;