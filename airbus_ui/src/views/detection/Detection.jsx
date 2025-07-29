// pages/AppDetection.jsx - Updated with 4-state system
import React, { useState, useEffect, useCallback, useRef } from "react";
import { 
  Box, 
  Grid, 
  Stack, 
  Alert, 
  CircularProgress, 
  Card,
  CardContent,
  Typography,
  Chip,
  Divider,
  Button
} from '@mui/material';

import DetectionControls from "./components/DetectionControls";
import DetectionVideoFeed from "./components/DetectionVideoFeed";
import { cameraService } from "../captureImage/CameraService";
import { detectionService } from "./detectionService";

// Detection states from service
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export default function AppDetection() {
  // Core state management
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);
  
  // Detection service state
  const [detectionState, setDetectionState] = useState(DetectionStates.INITIALIZING);
  const [initializationError, setInitializationError] = useState(null);
  const [systemHealth, setSystemHealth] = useState({
    streaming: { status: 'unknown' },
    detection: { status: 'unknown' },
    overall: false
  });

  // Performance and optimization state
  const [detectionOptions, setDetectionOptions] = useState({
    detectionFps: 5.0,
    streamQuality: 85,
    priority: 1,
    enableAdaptiveQuality: true,
    enableFrameSkipping: true
  });
  
  const [globalStats, setGlobalStats] = useState({
    totalStreams: 0,
    avgProcessingTime: 0,
    totalDetections: 0,
    systemLoad: 0,
    memoryUsage: 0
  });

  // Refs for lifecycle management
  const statsInterval = useRef(null);
  const cleanupRef = useRef(false);
  const initializationAttempted = useRef(false);
  const lastHealthCheck = useRef(null);
  const stateChangeUnsubscribe = useRef(null);

  // Subscribe to detection service state changes
  useEffect(() => {
    const unsubscribe = detectionService.addStateChangeListener((newState, oldState) => {
      console.log(`🔄 Detection state changed: ${oldState} → ${newState}`);
      setDetectionState(newState);
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
    
    // Get initial state
    setDetectionState(detectionService.getState());
    
    return () => {
      if (stateChangeUnsubscribe.current) {
        stateChangeUnsubscribe.current();
      }
    };
  }, []);

  // Initialize system on component mount
  useEffect(() => {
    const initializeSystem = async () => {
      if (initializationAttempted.current) return;
      initializationAttempted.current = true;

      try {
        console.log("🚀 Starting system initialization...");
        
        // Initialize detection processor
        const initResult = await detectionService.ensureInitialized();
        
        if (initResult.success) {
          console.log("✅ Detection system initialized:", initResult.message);
          setInitializationError(null);
          
          // Perform single health check
          await performSingleHealthCheck();
          
          // Start stats monitoring
          startStatsMonitoring();
          
          console.log("✅ System initialization completed successfully");
        } else {
          throw new Error(initResult.message || 'Failed to initialize detection system');
        }
        
      } catch (error) {
        console.error("❌ Error initializing system:", error);
        setInitializationError(error.message);
      }
    };

    // Only initialize if in INITIALIZING state
    if (detectionState === DetectionStates.INITIALIZING) {
      initializeSystem();
    }
    
    return () => {
      stopMonitoring();
    };
  }, [detectionState]);

  // Single health check function
  const performSingleHealthCheck = useCallback(async () => {
    if (detectionState === DetectionStates.SHUTTING_DOWN) {
      console.log("⏭️ Skipping health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing health check...");
      const health = await detectionService.checkOptimizedHealth();
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      if (!health.overall) {
        console.warn("System health check failed:", health);
      }
      
      console.log("✅ Health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
    }
  }, [detectionState]);

  // Manual retry initialization
  const retryInitialization = useCallback(async () => {
    initializationAttempted.current = false;
    setInitializationError(null);
    
    try {
      // Reset service to initializing state
      detectionService.resetToInitializing('Manual retry');
      
      // Wait a moment for state to update
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // The useEffect will trigger initialization automatically
      console.log("🔄 Retry initialization requested");
      
    } catch (error) {
      console.error("❌ Error during retry initialization:", error);
      setInitializationError(error.message);
    }
  }, []);

  // Stats monitoring (reduced frequency)
  const startStatsMonitoring = useCallback(() => {
    if (statsInterval.current) {
      clearInterval(statsInterval.current);
    }
    
    const updateGlobalStats = async () => {
      if (detectionState === DetectionStates.SHUTTING_DOWN) {
        console.log("⏭️ Skipping stats update - system is shutting down");
        return;
      }

      try {
        const stats = await detectionService.getAllStreamingStats();
        
        setGlobalStats({
          totalStreams: stats.active_streams || 0,
          avgProcessingTime: stats.avg_processing_time_ms || 0,
          totalDetections: stats.total_detections || 0,
          systemLoad: stats.system_load_percent || 0,
          memoryUsage: stats.memory_usage_mb || 0
        });
      } catch (error) {
        console.debug("Error fetching global stats:", error);
      }
    };

    updateGlobalStats();
    statsInterval.current = setInterval(updateGlobalStats, 10000);
  }, [detectionState]);

  // Stop all monitoring
  const stopMonitoring = useCallback(() => {
    console.log("🛑 Stopping all monitoring...");
    
    if (statsInterval.current) {
      clearInterval(statsInterval.current);
      statsInterval.current = null;
    }
  }, []);

  // Enhanced cleanup with proper shutdown signaling
  useEffect(() => {
    const handleBeforeUnload = async (event) => {
      if (cleanupRef.current) return;
      cleanupRef.current = true;
      
      try {
        console.log("Performing cleanup...");
        
        // Stop all monitoring first
        stopMonitoring();
        
        // Stop all detection services
        if (detectionState === DetectionStates.RUNNING) {
          await detectionService.stopAllStreams();
        }
        
        // Stop camera services
        await cameraService.stopCamera();
        await cameraService.cleanupTempPhotos();
        
        console.log("Cleanup completed");
      } catch (error) {
        console.error("Error during cleanup:", error);
      }
    };
    
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      handleBeforeUnload();
    };
  }, [detectionState, stopMonitoring]);
  
  // Fetch available cameras with error handling
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const cameraData = await cameraService.getAllCameras();
        setCameras(cameraData);
        console.log(`Loaded ${cameraData.length} cameras`);
      } catch (error) {
        console.error("Error fetching cameras:", error);
        setCameras([]);
      }
    };
    
    fetchCameras();
  }, []);

  // Handle camera selection with validation
  const handleCameraChange = useCallback((event) => {
    const selectedCameraId = event.target.value;
    console.log("Camera selection changed:", selectedCameraId);
    
    const cameraExists = cameras.some(cam => cam.id.toString() === selectedCameraId.toString());
    if (cameraExists || selectedCameraId === '') {
      setSelectedCameraId(selectedCameraId);
      setCameraId(selectedCameraId);
    } else {
      console.warn("Selected camera not found in available cameras");
      alert("Selected camera is not available. Please choose a different camera.");
    }
  }, [cameras]);
  
  // Enhanced detection start with state validation
  const handleStartDetection = useCallback(async () => {
    // Comprehensive validation
    if (!cameraId || cameraId === '') {
      alert("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      alert("Please enter a target label for detection.");
      return;
    }
    
    if (!detectionService.canStart()) {
      alert(`Cannot start detection. Current state: ${detectionState}. System must be READY.`);
      return;
    }
    
    // Perform health check before starting detection
    console.log("🩺 Checking system health before starting detection...");
    await performSingleHealthCheck();
    
    if (!systemHealth.overall) {
      const proceed = window.confirm(
        "System health check indicates issues. Do you want to proceed anyway?"
      );
      if (!proceed) return;
    }
    
    // Validate camera still exists
    const cameraExists = cameras.some(cam => cam.id.toString() === cameraId.toString());
    if (!cameraExists) {
      alert("Selected camera is no longer available. Please detect cameras and select a new one.");
      return;
    }
    
    console.log("Starting detection with options:", detectionOptions);
  }, [cameraId, targetLabel, detectionState, systemHealth.overall, cameras, detectionOptions, performSingleHealthCheck]);
  
  // Enhanced detection stop
  const handleStopDetection = useCallback(async () => {
    console.log("Stopping detection...");
    
    // The state will be managed by the service itself
    // Just trigger the stop, don't manage state here
    
    // Perform health check after shutdown to verify clean state
    setTimeout(async () => {
      console.log("🩺 Checking system health after detection stop...");
      await performSingleHealthCheck();
    }, 2000); // Wait 2 seconds for shutdown to complete
  }, [performSingleHealthCheck]);

  // Handle target label changes with validation
  const handleTargetLabelChange = useCallback((event) => {
    const value = event.target.value;
    const sanitizedValue = value.replace(/[<>"/\\&]/g, '');
    setTargetLabel(sanitizedValue);
  }, []);

  // Enhanced camera detection with better error handling
  const handleDetectCameras = useCallback(async () => {
    setIsDetecting(true);
    try {
      console.log("Detecting available cameras...");
      const detectedCameras = await cameraService.detectCameras();
      setCameras(detectedCameras);
      
      if (selectedCameraId && !detectedCameras.some(cam => cam.id.toString() === selectedCameraId.toString())) {
        console.log("Previously selected camera no longer available, resetting selection");
        setSelectedCameraId('');
        setCameraId('');
        
        if (detectionState === DetectionStates.RUNNING) {
          alert("The camera currently in use is no longer available. Detection has been stopped.");
          // The service will handle state transitions
        }
      }
      
      console.log(`Successfully detected ${detectedCameras.length} cameras`);
    } catch (error) {
      console.error("Error detecting cameras:", error);
      alert(`Camera detection failed: ${error.message}`);
    } finally {
      setIsDetecting(false);
    }
  }, [selectedCameraId, detectionState]);

  // Handle detection options changes
  const handleDetectionOptionsChange = useCallback((newOptions) => {
    setDetectionOptions(prev => ({
      ...prev,
      ...newOptions
    }));
    console.log("Detection options updated:", newOptions);
  }, []);

  // Manual health check button handler
  const handleManualHealthCheck = useCallback(async () => {
    console.log("🩺 Manual health check requested...");
    await performSingleHealthCheck();
  }, [performSingleHealthCheck]);

  // Performance status color helper
  const getPerformanceColor = (value, thresholds) => {
    if (value < thresholds.good) return 'success';
    if (value < thresholds.warning) return 'warning';
    return 'error';
  };

  // Helper to show when last health check was performed
  const getHealthCheckAge = () => {
    if (!lastHealthCheck.current) return 'Never';
    const ageMs = Date.now() - lastHealthCheck.current;
    const ageSeconds = Math.floor(ageMs / 1000);
    if (ageSeconds < 60) return `${ageSeconds}s ago`;
    const ageMinutes = Math.floor(ageSeconds / 60);
    return `${ageMinutes}m ago`;
  };

  // Get state-specific styling and messages
  const getStateInfo = () => {
    switch (detectionState) {
      case DetectionStates.INITIALIZING:
        return {
          color: 'info',
          message: 'Initializing detection system...',
          canOperate: false
        };
      case DetectionStates.READY:
        return {
          color: 'success',
          message: 'System ready for detection',
          canOperate: true
        };
      case DetectionStates.RUNNING:
        return {
          color: 'warning',
          message: 'Detection active',
          canOperate: true
        };
      case DetectionStates.SHUTTING_DOWN:
        return {
          color: 'error',
          message: 'System shutting down...',
          canOperate: false
        };
      default:
        return {
          color: 'default',
          message: 'Unknown state',
          canOperate: false
        };
    }
  };

  const stateInfo = getStateInfo();

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* State Status Alert */}
      {detectionState === DetectionStates.INITIALIZING && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing detection system... This may take a moment.
        </Alert>
      )}

      {detectionState === DetectionStates.SHUTTING_DOWN && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          System is shutting down... Please wait.
        </Alert>
      )}

      {/* System Status Alerts */}
      {initializationError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          System initialization failed: {initializationError}
          <Box sx={{ mt: 1 }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={retryInitialization}
              disabled={detectionState === DetectionStates.INITIALIZING || detectionState === DetectionStates.SHUTTING_DOWN}
            >
              {detectionState === DetectionStates.INITIALIZING ? 'Initializing...' : 'Retry Initialization'}
            </Button>
          </Box>
          <br />
          <small>If the issue persists, try refreshing the page or contact support.</small>
        </Alert>
      )}
      
      {!systemHealth.overall && detectionState === DetectionStates.READY && !initializationError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          System health check indicates issues. Detection may not work optimally.
          <br />
          <small>
            Streaming: {systemHealth.streaming.status} | 
            Detection: {systemHealth.detection.status} | 
            Last checked: {getHealthCheckAge()}
          </small>
          <Box sx={{ mt: 1 }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={handleManualHealthCheck}
              disabled={detectionState !== DetectionStates.READY}
            >
              Check Health Now
            </Button>
          </Box>
        </Alert>
      )}
      
      <Grid container spacing={2} columns={12} sx={{ mb: 2 }}>
        {/* Main Content */}
        <Grid size={{ xs: 12, md: 9 }}>
          <Stack spacing={3}>
            <DetectionControls
              targetLabel={targetLabel}
              onTargetLabelChange={handleTargetLabelChange}
              selectedCameraId={selectedCameraId}
              onCameraChange={handleCameraChange}
              cameras={cameras}
              onDetectCameras={handleDetectCameras}
              isDetecting={isDetecting}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              isDetectionActive={detectionState === DetectionStates.RUNNING}
              isSystemReady={stateInfo.canOperate && detectionState === DetectionStates.READY}
              systemHealth={systemHealth}
              detectionOptions={detectionOptions}
              onDetectionOptionsChange={handleDetectionOptionsChange}
              detectionState={detectionState}
            />
            
            <DetectionVideoFeed
              isDetectionActive={detectionState === DetectionStates.RUNNING}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              cameraId={cameraId}
              targetLabel={targetLabel}
              isSystemReady={stateInfo.canOperate}
              detectionOptions={detectionOptions}
              detectionState={detectionState}
            />
          </Stack>
        </Grid>

        {/* System Performance Panel */}
        <Grid size={{ xs: 12, md: 3 }}>
          <Card sx={{ height: 'fit-content' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance
              </Typography>
              
              <Stack spacing={2}>
                {/* Detection State */}
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Detection State
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                    <Chip
                      label={detectionState}
                      color={stateInfo.color}
                      size="small"
                      icon={
                        (detectionState === DetectionStates.INITIALIZING || detectionState === DetectionStates.SHUTTING_DOWN) ? 
                        <CircularProgress size={16} /> : undefined
                      }
                    />
                  </Stack>
                  <Typography variant="caption" color="textSecondary">
                    {stateInfo.message}
                  </Typography>
                </Box>

                {/* System Health */}
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    System Health
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                    <Chip
                      label={
                        detectionState === DetectionStates.INITIALIZING ? "Initializing" : 
                        detectionState === DetectionStates.SHUTTING_DOWN ? "Shutting Down" :
                        systemHealth.overall ? "Healthy" : "Issues Detected"
                      }
                      color={
                        detectionState === DetectionStates.INITIALIZING ? "warning" :
                        detectionState === DetectionStates.SHUTTING_DOWN ? "info" :
                        systemHealth.overall ? "success" : "error"
                      }
                      size="small"
                    />
                    <Button
                      size="small"
                      variant="text"
                      onClick={handleManualHealthCheck}
                      disabled={detectionState !== DetectionStates.READY}
                      sx={{ fontSize: '0.7rem', minWidth: 'auto', p: 0.5 }}
                    >
                      Check Now
                    </Button>
                  </Stack>
                  <Typography variant="caption" color="textSecondary">
                    Last checked: {getHealthCheckAge()}
                  </Typography>
                </Box>

                <Divider />

                {/* Performance Metrics */}
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Active Streams
                  </Typography>
                  <Typography variant="h4" color="primary">
                    {globalStats.totalStreams}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Avg Processing Time
                  </Typography>
                  <Chip
                    label={`${globalStats.avgProcessingTime.toFixed(1)}ms`}
                    color={getPerformanceColor(globalStats.avgProcessingTime, { good: 50, warning: 100 })}
                    size="small"
                  />
                </Box>

                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Total Detections
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {globalStats.totalDetections.toLocaleString()}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    System Load
                  </Typography>
                  <Chip
                    label={`${globalStats.systemLoad.toFixed(1)}%`}
                    color={getPerformanceColor(globalStats.systemLoad, { good: 60, warning: 80 })}
                    size="small"
                  />
                </Box>

                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Memory Usage
                  </Typography>
                  <Chip
                    label={`${globalStats.memoryUsage.toFixed(0)}MB`}
                    color={getPerformanceColor(globalStats.memoryUsage, { good: 1000, warning: 2000 })}
                    size="small"
                  />
                </Box>

                {/* Current Settings */}
                <Divider />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Detection Settings
                  </Typography>
                  <Stack spacing={0.5} sx={{ mt: 1 }}>
                    <Typography variant="caption">
                      FPS: {detectionOptions.detectionFps}
                    </Typography>
                    <Typography variant="caption">
                      Quality: {detectionOptions.streamQuality}%
                    </Typography>
                    <Typography variant="caption">
                      Priority: {detectionOptions.priority}
                    </Typography>
                  </Stack>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}