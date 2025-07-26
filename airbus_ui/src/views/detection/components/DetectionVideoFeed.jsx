// DetectionVideoFeed.jsx - Enhanced component with auto-initialization
import React, { useState, useEffect, useRef, useCallback } from "react";
import { Box, Alert, CircularProgress, Typography, Chip } from "@mui/material";
import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";
import { detectionService } from "../detectionService";

const DetectionVideoFeed = ({
  isDetectionActive,
  onStartDetection,
  onStopDetection,
  cameraId,
  targetLabel,
  isModelLoaded,
  onModelLoadedChange = () => {}, // Add fallback function to prevent errors
  detectionOptions = {}
}) => {
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
  const [initializationStatus, setInitializationStatus] = useState({
    isInitializing: false,
    initializationError: null,
    initializationAttempts: 0
  });
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  
  const videoRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const healthCheckIntervalRef = useRef(null);
  const initializationAttemptedRef = useRef(false);

  // Initialize detection processor when component mounts or when model loading is attempted
  useEffect(() => {
    const initializeDetectionSystem = async () => {
      // Prevent multiple initialization attempts
      if (initializationAttemptedRef.current || initializationStatus.isInitializing) {
        return;
      }

      initializationAttemptedRef.current = true;
      setInitializationStatus(prev => ({ 
        ...prev, 
        isInitializing: true, 
        initializationError: null 
      }));

      try {
        console.log("Initializing detection system...");
        
        // First ensure the processor is initialized
        const initResult = await detectionService.ensureInitialized();
        console.log("Processor initialization result:", initResult);

        // Then load/check the model
        const modelResult = await detectionService.loadModel();
        console.log("Model loading result:", modelResult);
        
        // Notify parent component about model loading status
        if (typeof onModelLoadedChange === 'function') {
          onModelLoadedChange(modelResult.success);
        }
        
        setInitializationStatus(prev => ({ 
          ...prev, 
          isInitializing: false,
          initializationError: null
        }));

        console.log("Detection system initialized successfully");

      } catch (error) {
        console.error("Failed to initialize detection system:", error);
        
        setInitializationStatus(prev => ({ 
          ...prev, 
          isInitializing: false,
          initializationError: error.message,
          initializationAttempts: prev.initializationAttempts + 1
        }));
        
        // Notify parent component about failure
        if (typeof onModelLoadedChange === 'function') {
          onModelLoadedChange(false);
        }
        
        // Reset the attempt flag to allow retry
        initializationAttemptedRef.current = false;
      }
    };

    // Initialize on component mount
    initializeDetectionSystem();
  }, []); // Empty dependency array - run once on mount

  // Retry initialization function
  const retryInitialization = useCallback(async () => {
    console.log("Retrying detection system initialization...");
    initializationAttemptedRef.current = false;
    setInitializationStatus(prev => ({ 
      ...prev, 
      initializationError: null 
    }));
    
    // Re-run initialization
    const initializeDetectionSystem = async () => {
      if (initializationAttemptedRef.current || initializationStatus.isInitializing) {
        return;
      }

      initializationAttemptedRef.current = true;
      setInitializationStatus(prev => ({ 
        ...prev, 
        isInitializing: true 
      }));

      try {
        const initResult = await detectionService.ensureInitialized();
        const modelResult = await detectionService.loadModel();
        
        // Notify parent component
        if (typeof onModelLoadedChange === 'function') {
          onModelLoadedChange(modelResult.success);
        }
        
        setInitializationStatus(prev => ({ 
          ...prev, 
          isInitializing: false,
          initializationError: null
        }));

      } catch (error) {
        console.error("Retry failed:", error);
        
        setInitializationStatus(prev => ({ 
          ...prev, 
          isInitializing: false,
          initializationError: error.message,
          initializationAttempts: prev.initializationAttempts + 1
        }));
        
        // Notify parent component about failure
        if (typeof onModelLoadedChange === 'function') {
          onModelLoadedChange(false);
        }
        
        initializationAttemptedRef.current = false;
      }
    };

    await initializeDetectionSystem();
  }, [initializationStatus.isInitializing, onModelLoadedChange]);

  // Stats listener callback
  const handleStatsUpdate = useCallback((newStats) => {
    setDetectionStats(prevStats => ({
      ...prevStats,
      ...newStats,
      detectionCount: newStats.objectDetected && !prevStats.objectDetected 
        ? prevStats.detectionCount + 1 
        : (newStats.detectionCount || prevStats.detectionCount)
    }));
  }, []);

  // Monitor video stream health
  const monitorStreamHealth = useCallback(() => {
    if (videoRef.current && isDetectionActive) {
      const video = videoRef.current;
      
      const checkVideoHealth = () => {
        if (video.readyState >= 2) {
          setStreamStatus(prev => ({ ...prev, isConnected: true, error: null }));
        } else if (video.networkState === 3) {
          setStreamStatus(prev => ({ 
            ...prev, 
            isConnected: false, 
            error: "Stream connection lost" 
          }));
        }
      };

      video.addEventListener('loadstart', () => {
        setStreamStatus(prev => ({ ...prev, isLoading: true }));
      });

      video.addEventListener('loadeddata', () => {
        setStreamStatus(prev => ({ ...prev, isLoading: false, isConnected: true }));
      });

      video.addEventListener('error', (e) => {
        console.error("Video stream error:", e);
        setStreamStatus(prev => ({ 
          ...prev, 
          isLoading: false, 
          isConnected: false, 
          error: "Stream error occurred" 
        }));
      });

      video.addEventListener('stalled', () => {
        setStreamStatus(prev => ({ ...prev, error: "Stream stalled" }));
      });

      const healthInterval = setInterval(checkVideoHealth, 5000);
      healthCheckIntervalRef.current = healthInterval;

      return () => {
        clearInterval(healthInterval);
        video.removeEventListener('loadstart', () => {});
        video.removeEventListener('loadeddata', () => {});
        video.removeEventListener('error', () => {});
        video.removeEventListener('stalled', () => {});
      };
    }
  }, [isDetectionActive]);

  // Enhanced detection start with proper initialization check
  const handleStartDetection = async () => {
    if (!cameraId || cameraId === '') {
      alert("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      alert("Please enter a target label first.");
      return;
    }

    // Check if initialization is in progress
    if (initializationStatus.isInitializing) {
      alert("Detection system is still initializing. Please wait...");
      return;
    }

    // If there's an initialization error, offer to retry
    if (initializationStatus.initializationError) {
      const shouldRetry = window.confirm(
        `Detection system initialization failed: ${initializationStatus.initializationError}\n\nWould you like to retry initialization?`
      );
      if (shouldRetry) {
        await retryInitialization();
        return; // Exit here, user can try starting detection again after retry
      } else {
        return;
      }
    }
    
    if (!isModelLoaded) {
      alert("Detection model is not loaded. Please wait for initialization to complete or try refreshing the page.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });

    try {
      // Double-check that the system is properly initialized before starting
      await detectionService.ensureInitialized();

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

      console.log(`Started optimized detection for camera ${cameraId} with target: ${targetLabel}`);

    } catch (error) {
      console.error("Error starting optimized detection:", error);
      
      // If the error suggests the processor isn't initialized, try to reinitialize
      if (error.message.includes('not initialized') || error.message.includes('503')) {
        console.log("Processor seems uninitialized, attempting reinitialization...");
        try {
          await retryInitialization();
          alert("Detection system was reinitialized. Please try starting detection again.");
        } catch (reinitError) {
          alert(`Failed to reinitialize detection system: ${reinitError.message}`);
        }
      } else {
        setStreamStatus({ 
          isLoading: false, 
          error: error.message || "Failed to start detection", 
          isConnected: false 
        });
        alert(`Failed to start detection: ${error.message}`);
      }
    }
  };

  // Handle stopping detection
  const handleStopDetection = async () => {
    setStreamStatus(prev => ({ ...prev, isLoading: true }));

    try {
      detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      await detectionService.stopOptimizedDetectionFeed(parseInt(cameraId));
      
      setVideoUrl("");
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
      
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      onStopDetection();

      console.log(`Stopped optimized detection for camera ${cameraId}`);

    } catch (error) {
      console.error("Error stopping optimized detection:", error);
      setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop detection" }));
      alert(`Failed to stop detection: ${error.message}`);
    }
  };

  // Load performance metrics periodically
  useEffect(() => {
    if (isDetectionActive) {
      const loadPerformanceMetrics = async () => {
        try {
          const metrics = await detectionService.getPerformanceComparison();
          setPerformanceMetrics(metrics);
        } catch (error) {
          console.debug("Error loading performance metrics:", error);
        }
      };

      loadPerformanceMetrics();
      const metricsInterval = setInterval(loadPerformanceMetrics, 10000);
      return () => clearInterval(metricsInterval);
    }
  }, [isDetectionActive]);

  // Setup video health monitoring
  useEffect(() => {
    return monitorStreamHealth();
  }, [monitorStreamHealth]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
      }
      if (cameraId) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
    };
  }, [cameraId, handleStatsUpdate]);

  // Auto-reconnect on stream failure
  useEffect(() => {
    if (streamStatus.error && isDetectionActive && !streamStatus.isLoading) {
      console.log("Stream error detected, attempting reconnection...");
      
      reconnectTimeoutRef.current = setTimeout(() => {
        handleStartDetection();
      }, 3000);
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [streamStatus.error, isDetectionActive, streamStatus.isLoading]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* Initialization Status Alerts */}
      {initializationStatus.isInitializing && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing detection system... This may take a few moments.
        </Alert>
      )}

      {initializationStatus.initializationError && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, width: '100%' }}
          action={
            <button 
              onClick={retryInitialization}
              style={{ 
                background: 'none', 
                border: '1px solid currentColor', 
                color: 'inherit',
                padding: '4px 8px',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          }
        >
          Detection system initialization failed: {initializationStatus.initializationError}
          {initializationStatus.initializationAttempts > 1 && 
            ` (Attempt ${initializationStatus.initializationAttempts})`
          }
        </Alert>
      )}

      {/* Stream Status Alerts */}
      {streamStatus.error && (
        <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
          {streamStatus.error}
          {isDetectionActive && " - Attempting to reconnect..."}
        </Alert>
      )}

      {streamStatus.isLoading && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {isDetectionActive ? "Connecting to optimized detection stream..." : "Stopping stream..."}
        </Alert>
      )}

      {/* Performance Metrics Display */}
      {performanceMetrics && isDetectionActive && (
        <Box sx={{ mb: 2, width: '100%' }}>
          <Typography variant="caption" color="textSecondary">
            Performance Metrics:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
            <Chip 
              label={`Avg Processing: ${detectionStats.avgProcessingTime.toFixed(1)}ms`} 
              size="small" 
              color={detectionStats.avgProcessingTime < 100 ? "success" : "warning"}
            />
            <Chip 
              label={`Detection FPS: ${detectionStats.detectionFps.toFixed(1)}`} 
              size="small" 
              color="info"
            />
            <Chip 
              label={`Queue Depth: ${detectionStats.queueDepth}`} 
              size="small" 
              color={detectionStats.queueDepth < 3 ? "success" : "warning"}
            />
            <Chip 
              label={`Quality: ${detectionStats.streamQuality}%`} 
              size="small" 
              color="default"
            />
          </Box>
        </Box>
      )}

      <VideoCard
        cameraActive={isDetectionActive}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isDetectionActive ? (
          <LiveDetectionView
            videoUrl={videoUrl}
            videoRef={videoRef}
            showControls={showControls}
            onStopDetection={handleStopDetection}
            detectionStats={detectionStats}
            targetLabel={targetLabel}
            streamStatus={streamStatus}
            performanceMetrics={performanceMetrics}
          />
        ) : (
          <CameraPlaceholder 
            onStartCamera={handleStartDetection}
            cameraId={cameraId}
            buttonText={
              initializationStatus.isInitializing 
                ? "Initializing..." 
                : initializationStatus.initializationError 
                  ? "Retry Initialization" 
                  : "Start Optimized Detection"
            }
            icon="detection"
            disabled={
              initializationStatus.isInitializing ||
              !targetLabel || 
              !cameraId || 
              streamStatus.isLoading ||
              (!isModelLoaded && !initializationStatus.initializationError)
            }
            isLoading={initializationStatus.isInitializing || streamStatus.isLoading}
          />
        )}
      </VideoCard>

      {/* Debug Information (Development only) */}
      {process.env.NODE_ENV === 'development' && (
        <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.100', borderRadius: 1, width: '100%' }}>
          <Typography variant="caption" color="textSecondary">
            Debug Info:
          </Typography>
          <pre style={{ fontSize: '0.7rem', margin: 0 }}>
            {JSON.stringify({
              cameraId: parseInt(cameraId),
              targetLabel,
              isModelLoaded,
              initializationStatus: {
                isInitializing: initializationStatus.isInitializing,
                hasError: !!initializationStatus.initializationError,
                attempts: initializationStatus.initializationAttempts
              },
              streamConnected: streamStatus.isConnected,
              detectionStats: {
                detected: detectionStats.objectDetected,
                count: detectionStats.detectionCount,
                avgTime: detectionStats.avgProcessingTime,
                queueDepth: detectionStats.queueDepth
              }
            }, null, 2)}
          </pre>
        </Box>
      )}
    </div>
  );
};

export default DetectionVideoFeed;