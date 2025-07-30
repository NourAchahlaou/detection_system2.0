// DetectionVideoFeed.jsx - Enhanced with Basic Mode On-Demand Detection
import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  Box, 
  Alert, 
  CircularProgress, 
  Typography, 
  Chip, 
  Button,
  Stack,
  Paper,
  IconButton,
  Tooltip,
  Badge
} from "@mui/material";
import { 
  PlayArrow, 
  Pause, 
  CameraAlt, 
  Visibility, 
  Refresh,
  FiberManualRecord,
  Stop
} from '@mui/icons-material';
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
    isStreamActive: false,
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
  const [detectionInProgress, setDetectionInProgress] = useState(false);
  const [lastDetectionResult, setLastDetectionResult] = useState(null);
  const [frozenFrame, setFrozenFrame] = useState(null);
  const [basicModeStats, setBasicModeStats] = useState({
    totalDetections: 0,
    successfulDetections: 0,
    lastDetectionTime: null,
    avgProcessingTime: 0
  });
  
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
  const initializationAttempted = useRef(false);
  const basicModeStatsInterval = useRef(null);

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
          resetBasicModeState();
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
            resetBasicModeState();
            
            setHealthCheckStatus(prev => ({
              ...prev,
              postShutdown: false
            }));
          } else if (oldState === DetectionStates.INITIALIZING) {
            setStreamStatus(prev => ({ ...prev, isLoading: false }));
            setComponentError(null);
            
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

  // Basic mode state monitoring
  useEffect(() => {
    if (isBasicMode && isDetectionActive && cameraId) {
      const monitorBasicModeState = async () => {
        try {
          const streamFrozen = detectionService.isStreamFrozen(cameraId);
          const detectionActive = detectionService.isDetectionInProgress(cameraId);
          const lastResult = detectionService.getLastDetectionResult(cameraId);
          const frozen = detectionService.getFrozenFrame(cameraId);
          
          setIsStreamFrozen(streamFrozen);
          setDetectionInProgress(detectionActive);
          setLastDetectionResult(lastResult);
          setFrozenFrame(frozen);
          
          // Update basic mode stats
          if (lastResult) {
            setBasicModeStats(prev => ({
              ...prev,
              lastDetectionTime: Date.now(),
              avgProcessingTime: lastResult.processing_time_ms || prev.avgProcessingTime
            }));
          }
        } catch (error) {
          console.error('Error monitoring basic mode state:', error);
        }
      };

      // Monitor every 1 second
      basicModeStatsInterval.current = setInterval(monitorBasicModeState, 1000);
      
      return () => {
        if (basicModeStatsInterval.current) {
          clearInterval(basicModeStatsInterval.current);
        }
      };
    }
  }, [isBasicMode, isDetectionActive, cameraId]);

  // Reset functions
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
      mode: currentMode
    });
  }, [currentMode]);

  const resetBasicModeState = useCallback(() => {
    setIsStreamFrozen(false);
    setDetectionInProgress(false);
    setLastDetectionResult(null);
    setFrozenFrame(null);
    setBasicModeStats({
      totalDetections: 0,
      successfulDetections: 0,
      lastDetectionTime: null,
      avgProcessingTime: 0
    });
  }, []);

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

  // Health check functions
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckStatus.initial) {
      console.log("⏭️ VideoFeed: Initial health check already performed");
      return;
    }

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

  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckStatus.postShutdown) {
      console.log("⏭️ VideoFeed: Post-shutdown health check already performed");
      return;
    }

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

  // Enhanced detection start with mode awareness
  const handleStartDetection = async () => {
    if (!cameraId || cameraId === '') {
      setComponentError("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      setComponentError("Please enter a target label first.");
      return;
    }

    const currentState = detectionService.getState();
    console.log(`🎯 VideoFeed: Attempting to start detection. Current state: ${currentState}, Mode: ${currentMode}`);

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
    setModeTransitioning(true);
    resetBasicModeState();

    try {
      let streamUrl;
      
      if (isBasicMode) {
        // Start basic mode with on-demand detection
        console.log(`🎯 Starting basic mode stream for camera ${cameraId}`);
        streamUrl = await detectionService.startBasicDetectionFeed(
          parseInt(cameraId), 
          targetLabel, 
          {
            streamQuality: detectionOptions.streamQuality || 85
          }
        );
      } else {
        // Start optimized mode with continuous detection
        streamUrl = await detectionService.startDetectionFeedWithAutoMode(
          parseInt(cameraId), 
          targetLabel, 
          {
            detectionFps: detectionOptions.detectionFps || 5.0,
            streamQuality: detectionOptions.streamQuality || 85,
            priority: detectionOptions.priority || 1
          }
        );
        
        // Add stats listener for optimized mode
        detectionService.addStatsListener(parseInt(cameraId), handleStatsUpdate);
      }

      if (!mountedRef.current) return;

      setVideoUrl(streamUrl);
      
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
        mode: currentMode
      });

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      setModeTransitioning(false);
      onStartDetection();

      console.log(`✅ VideoFeed: Started ${currentMode} detection for camera ${cameraId} with target: ${targetLabel}`);

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
    
    console.log(`🛑 VideoFeed: Stopping ${currentMode} detection for camera ${cameraId}`);
    setStreamStatus(prev => ({ ...prev, isLoading: true }));
    setModeTransitioning(true);

    try {
      // Remove stats listener for optimized mode
      if (!isBasicMode) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
      
      await detectionService.stopDetectionFeed(parseInt(cameraId), false);
      
      if (!mountedRef.current) return;
      
      setVideoUrl("");
      resetDetectionStats();
      resetBasicModeState();
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      setComponentError(null);
      setModeTransitioning(false);
      onStopDetection();

      console.log(`✅ VideoFeed: Stopped ${currentMode} detection for camera ${cameraId}`);

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

  // Basic Mode: Perform on-demand detection
  const handlePerformDetection = async () => {
    if (!isBasicMode || !cameraId || !targetLabel) {
      console.error('Cannot perform detection: missing requirements');
      return;
    }

    if (detectionInProgress) {
      console.log('Detection already in progress, skipping...');
      return;
    }

    setDetectionInProgress(true);
    setComponentError(null);

    try {
      console.log(`🎯 Performing on-demand detection for camera ${cameraId} with target: ${targetLabel}`);
      
      const result = await detectionService.performOnDemandDetection(
        parseInt(cameraId),
        targetLabel,
        {
          quality: detectionOptions.streamQuality || 85,
          autoUnfreeze: false, // Keep stream frozen to show results
          unfreezeDelay: 2.0
        }
      );

      if (result.success && mountedRef.current) {
        setLastDetectionResult(result.data);
        setFrozenFrame(result.data.annotated_image_base64);
        setIsStreamFrozen(true);
        
        // Update basic mode stats
        setBasicModeStats(prev => ({
          totalDetections: prev.totalDetections + 1,
          successfulDetections: result.data.target_detected ? prev.successfulDetections + 1 : prev.successfulDetections,
          lastDetectionTime: Date.now(),
          avgProcessingTime: result.data.processing_time_ms || prev.avgProcessingTime
        }));

        // Update main detection stats
        setDetectionStats(prev => ({
          ...prev,
          objectDetected: result.data.target_detected,
          detectionCount: result.data.target_detected ? prev.detectionCount + 1 : prev.detectionCount,
          lastDetectionTime: Date.now(),
          avgProcessingTime: result.data.processing_time_ms || prev.avgProcessingTime
        }));

        console.log(`✅ Detection completed. Target detected: ${result.data.target_detected}, Confidence: ${result.data.confidence}`);
      }

    } catch (error) {
      console.error('❌ Error performing on-demand detection:', error);
      setComponentError(`Detection failed: ${error.message}`);
    } finally {
      setDetectionInProgress(false);
    }
  };

  // Basic Mode: Unfreeze stream
  const handleUnfreezeStream = async () => {
    if (!isBasicMode || !cameraId) {
      return;
    }

    try {
      console.log(`🔥 Unfreezing stream for camera ${cameraId}`);
      await detectionService.unfreezeBasicStream(parseInt(cameraId));
      
      setIsStreamFrozen(false);
      setFrozenFrame(null);
      // Keep last detection result for reference
      
      console.log('✅ Stream unfrozen successfully');
    } catch (error) {
      console.error('❌ Error unfreezing stream:', error);
      setComponentError(`Failed to unfreeze stream: ${error.message}`);
    }
  };

  // Basic Mode: Detect and auto-resume
  const handleDetectAndResume = async () => {
    if (!isBasicMode || !cameraId || !targetLabel) {
      console.error('Cannot perform detection: missing requirements');
      return;
    }

    if (detectionInProgress) {
      console.log('Detection already in progress, skipping...');
      return;
    }

    setDetectionInProgress(true);
    setComponentError(null);

    try {
      console.log(`🎯 Performing detection with auto-resume for camera ${cameraId}`);
      
      const result = await detectionService.performOnDemandDetection(
        parseInt(cameraId),
        targetLabel,
        {
          quality: detectionOptions.streamQuality || 85,
          autoUnfreeze: true, // Auto-resume after delay
          unfreezeDelay: 3.0 // Show results for 3 seconds
        }
      );

      if (result.success && mountedRef.current) {
        setLastDetectionResult(result.data);
        setFrozenFrame(result.data.annotated_image_base64);
        setIsStreamFrozen(!result.autoUnfreeze); // Stream will be unfrozen automatically
        
        // Update stats
        setBasicModeStats(prev => ({
          totalDetections: prev.totalDetections + 1,
          successfulDetections: result.data.target_detected ? prev.successfulDetections + 1 : prev.successfulDetections,
          lastDetectionTime: Date.now(),
          avgProcessingTime: result.data.processing_time_ms || prev.avgProcessingTime
        }));

        setDetectionStats(prev => ({
          ...prev,
          objectDetected: result.data.target_detected,
          detectionCount: result.data.target_detected ? prev.detectionCount + 1 : prev.detectionCount,
          lastDetectionTime: Date.now(),
          avgProcessingTime: result.data.processing_time_ms || prev.avgProcessingTime
        }));

        // Auto-clear frozen frame after delay if auto-unfrozen
        if (result.autoUnfreeze) {
          setTimeout(() => {
            if (mountedRef.current) {
              setIsStreamFrozen(false);
              setFrozenFrame(null);
            }
          }, 3500); // Slightly longer than unfreeze delay
        }

        console.log(`✅ Detection with auto-resume completed. Target detected: ${result.data.target_detected}`);
      }

    } catch (error) {
      console.error('❌ Error performing detection with auto-resume:', error);
      setComponentError(`Detection failed: ${error.message}`);
    } finally {
      setDetectionInProgress(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraId && !isBasicMode) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
      if (basicModeStatsInterval.current) {
        clearInterval(basicModeStatsInterval.current);
      }
    };
  }, [cameraId, isBasicMode, handleStatsUpdate]);

  // Get appropriate button text based on state and mode
  const getButtonText = () => {
    if (modeTransitioning) return "Switching Mode...";
    
    switch (detectionState) {
      case DetectionStates.INITIALIZING:
        return "Initializing System...";
      case DetectionStates.READY:
        return componentError ? "Retry" : `Start ${isBasicMode ? 'Basic' : 'Optimized'} Detection`;
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

  // Enhanced Basic Mode Controls Component
  const BasicModeControls = () => {
    if (!isBasicMode || !isDetectionActive) return null;

    return (
      <Paper 
        elevation={3} 
        sx={{ 
          position: 'absolute', 
          bottom: 16, 
          left: 16, 
          right: 16, 
          p: 2, 
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          zIndex: 10
        }}
      >
        <Stack spacing={2}>
          {/* Status Info */}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Badge color={isStreamFrozen ? "error" : "success"} variant="dot">
                <FiberManualRecord fontSize="small" />
              </Badge>
              {isStreamFrozen ? 'Stream Frozen' : 'Live Stream'}
            </Typography>
            
            <Stack direction="row" spacing={1}>
              <Chip 
                label={`Detections: ${basicModeStats.totalDetections}`} 
                size="small" 
                color="primary"
              />
              <Chip 
                label={`Success: ${basicModeStats.successfulDetections}`} 
                size="small" 
                color="success"
              />
            </Stack>
          </Box>

          {/* Action Buttons */}
          <Stack direction="row" spacing={1} justifyContent="center">
            <Button
              variant="contained"
              color="primary"
              onClick={handlePerformDetection}
              disabled={detectionInProgress}
              startIcon={detectionInProgress ? <CircularProgress size={16} color="inherit" /> : <CameraAlt />}
              size="small"
            >
              {detectionInProgress ? 'Detecting...' : 'Detect Object'}
            </Button>

            <Button
              variant="outlined"
              color="secondary"
              onClick={handleDetectAndResume}
              disabled={detectionInProgress}
              startIcon={<Visibility />}
              size="small"
            >
              Detect & Resume
            </Button>

            {isStreamFrozen && (
              <Button
                variant="outlined"
                color="success"
                onClick={handleUnfreezeStream}
                startIcon={<PlayArrow />}
                size="small"
              >
                Resume Stream
              </Button>
            )}

            <Tooltip title="Stop Detection">
              <IconButton
                color="error"
                onClick={handleStopDetection}
                size="small"
              >
                <Stop />
              </IconButton>
            </Tooltip>
          </Stack>

          {/* Detection Results Display */}
          {lastDetectionResult && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="textSecondary">
                Last Detection: {lastDetectionResult.target_detected ? '✅ Found' : '❌ Not Found'} | 
                Confidence: {(lastDetectionResult.confidence * 100).toFixed(1)}% | 
                Processing: {lastDetectionResult.processing_time_ms}ms
              </Typography>
              
              {lastDetectionResult.detections && lastDetectionResult.detections.length > 0 && (
                <Typography variant="caption" display="block" color="success.main">
                  Objects detected: {lastDetectionResult.detections.length}
                </Typography>
              )}
            </Box>
          )}
        </Stack>
      </Paper>
    );
  };

  // Enhanced Live Detection View with Basic Mode Support
  const EnhancedLiveDetectionView = () => {
    if (!isDetectionActive || !videoUrl) return null;

    return (
      <Box sx={{ position: 'relative', width: '100%', height: '400px' }}>
        {/* Frozen Frame Overlay for Basic Mode */}
        {isBasicMode && isStreamFrozen && frozenFrame && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(0, 0, 0, 0.9)'
            }}
          >
            <img
              src={`data:image/jpeg;base64,${frozenFrame}`}
              alt="Detection Result"
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
                border: '2px solid #1976d2'
              }}
            />
            
            {/* Frozen Frame Indicator */}
            <Paper
              sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                p: 1,
                backgroundColor: 'rgba(255, 0, 0, 0.8)',
                color: 'white'
              }}
            >
              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Pause fontSize="small" />
                Detection Result
              </Typography>
            </Paper>
          </Box>
        )}

        {/* Live Video Stream */}
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
          // Basic mode specific props
          isStreamFrozen={isStreamFrozen}
          detectionInProgress={detectionInProgress}
          basicModeStats={basicModeStats}
        />

        {/* Basic Mode Controls Overlay */}
        <BasicModeControls />

        {/* Detection Progress Indicator for Basic Mode */}
        {isBasicMode && detectionInProgress && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 15,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
              p: 3,
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              borderRadius: 2,
              color: 'white'
            }}
          >
            <CircularProgress size={48} />
            <Typography variant="h6">Analyzing Frame...</Typography>
            <Typography variant="body2" color="textSecondary">
              Detecting {targetLabel}
            </Typography>
          </Box>
        )}
      </Box>
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
          {isDetectionActive ? `Connecting to ${currentMode} detection stream...` : "Stopping stream..."}
        </Alert>
      )}

      {/* Basic Mode Information Alert */}
      {isBasicMode && isDetectionActive && (
        <Alert severity="info" sx={{ mb: 2, width: '100%' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
            Basic Mode: On-Demand Detection Active
          </Typography>
          <Typography variant="body2">
            Stream is running continuously. Click "Detect Object" to analyze the current frame.
            The stream will freeze to show detection results.
          </Typography>
        </Alert>
      )}

      <VideoCard
        cameraActive={isDetectionActive && detectionState === DetectionStates.RUNNING}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isDetectionActive && detectionState === DetectionStates.RUNNING ? (
          <EnhancedLiveDetectionView />
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

      {/* Enhanced System Status Info with Basic Mode Features */}
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

        {/* Current Mode Display */}
        <Chip 
          label={`Mode: ${currentMode.toUpperCase()}`} 
          size="small" 
          color={isBasicMode ? 'warning' : 'success'}
        />

        {/* Basic Mode Specific Status */}
        {isBasicMode && isDetectionActive && (
          <>
            <Chip 
              label={isStreamFrozen ? 'Stream: Frozen' : 'Stream: Live'} 
              size="small" 
              color={isStreamFrozen ? 'error' : 'success'}
            />
            
            {detectionInProgress && (
              <Chip 
                label="Detecting..." 
                size="small" 
                color="info"
                icon={<CircularProgress size={12} color="inherit" />}
              />
            )}
            
            <Chip 
              label={`Detections: ${basicModeStats.totalDetections}`} 
              size="small" 
              color="primary"
            />
            
            {basicModeStats.successfulDetections > 0 && (
              <Chip 
                label={`Found: ${basicModeStats.successfulDetections}`} 
                size="small" 
                color="success"
              />
            )}
          </>
        )}

        {/* Auto Mode Status */}
        <Chip 
          label={detectionService.autoModeEnabled ? 'Auto Mode: ON' : 'Auto Mode: OFF'} 
          size="small" 
          color={detectionService.autoModeEnabled ? 'success' : 'default'}
        />

        {/* System Performance Score */}
        {systemProfile && (
          <Chip 
            label={`Score: ${systemProfile.performance_score}/100`} 
            size="small" 
            color={
              systemProfile.performance_score >= 70 ? 'success' :
              systemProfile.performance_score >= 40 ? 'warning' : 'error'
            }
          />
        )}
        
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

        {/* Processing Time for Basic Mode */}
        {isBasicMode && basicModeStats.avgProcessingTime > 0 && (
          <Chip 
            label={`Processing: ${basicModeStats.avgProcessingTime}ms`} 
            size="small" 
            color={basicModeStats.avgProcessingTime < 1000 ? 'success' : 'warning'}
          />
        )}
      </Box>

      {/* Quick Action Buttons for Basic Mode */}
      {isBasicMode && isDetectionActive && !isStreamFrozen && (
        <Box sx={{ mt: 1, width: '100%', display: 'flex', gap: 1, justifyContent: 'center' }}>
          <Button
            size="small"
            variant="contained"
            onClick={handlePerformDetection}
            disabled={detectionInProgress}
            startIcon={detectionInProgress ? <CircularProgress size={16} /> : <CameraAlt />}
          >
            {detectionInProgress ? 'Detecting...' : 'Detect Now'}
          </Button>
          
          <Button
            size="small"
            variant="outlined"
            onClick={handleDetectAndResume}
            disabled={detectionInProgress}
            startIcon={<Visibility />}
          >
            Quick Detect
          </Button>
        </Box>
      )}

      {/* Resume Controls for Frozen Stream */}
      {isBasicMode && isDetectionActive && isStreamFrozen && (
        <Box sx={{ mt: 1, width: '100%', display: 'flex', gap: 1, justifyContent: 'center' }}>
          <Button
            size="small"
            variant="contained"
            color="success"
            onClick={handleUnfreezeStream}
            startIcon={<PlayArrow />}
          >
            Resume Live Stream
          </Button>
          
          <Button
            size="small"
            variant="outlined"
            onClick={handlePerformDetection}
            disabled={detectionInProgress}
            startIcon={<Refresh />}
          >
            Detect Again
          </Button>
        </Box>
      )}
    </div>
  );
};

export default DetectionVideoFeed;