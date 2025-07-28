// pages/AppDetection.jsx - Fixed with smart health checking
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

export default function AppDetection() {
  // Core state management
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);
  
  // Model and system state
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [initializationStatus, setInitializationStatus] = useState('');
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

  // FIXED: Smart health checking refs
  const healthCheckInterval = useRef(null);
  const statsInterval = useRef(null);
  const cleanupRef = useRef(false);
  const initializationAttempted = useRef(false);
  const lastHealthCheck = useRef(null);
  const isShuttingDown = useRef(false);

  // Initialize system on component mount
  useEffect(() => {
    const initializeSystem = async () => {
      if (initializationAttempted.current) return;
      initializationAttempted.current = true;

      setIsInitializing(true);
      setInitializationStatus('Initializing detection processor...');
      
      try {
        console.log("Starting system initialization...");
        
        // Step 1: Initialize detection processor
        setInitializationStatus('Loading detection model...');
        const modelResult = await detectionService.loadModel();
        
        if (modelResult.success) {
          setIsModelLoaded(true);
          setInitializationStatus('Detection model loaded successfully');
          console.log("Detection model loaded:", modelResult.message);
        } else {
          throw new Error(modelResult.message || 'Failed to load detection model');
        }
        
        // Step 2: Check system health ONCE during initialization
        setInitializationStatus('Checking system health...');
        await performSingleHealthCheck();
        
        // Step 3: FIXED - Only start stats monitoring, no continuous health checks
        setInitializationStatus('Starting system monitoring...');
        startStatsMonitoring();
        
        setInitializationStatus('System ready');
        console.log("System initialization completed successfully");
        
      } catch (error) {
        console.error("Error initializing system:", error);
        setModelLoadError(`System initialization failed: ${error.message}`);
        setIsModelLoaded(false);
        setInitializationStatus(`Initialization failed: ${error.message}`);
      } finally {
        setIsInitializing(false);
      }
    };

    initializeSystem();
    
    return () => {
      stopMonitoring();
    };
  }, []);

  // FIXED: Single health check function (not continuous)
  const performSingleHealthCheck = useCallback(async () => {
    if (isShuttingDown.current) {
      console.log("⏭️ Skipping health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing single health check...");
      const health = await detectionService.checkOptimizedHealth();
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      if (!health.overall) {
        console.warn("System health check failed:", health);
        
        // If detection service is unhealthy, try to re-initialize
        if (health.detection.status === 'unhealthy' && !isShuttingDown.current) {
          console.log("Detection service unhealthy, attempting re-initialization...");
          try {
            await detectionService.ensureInitialized();
            // Retry health check after re-initialization
            const retryHealth = await detectionService.checkOptimizedHealth();
            setSystemHealth(retryHealth);
            lastHealthCheck.current = Date.now();
          } catch (reinitError) {
            console.error("Re-initialization failed:", reinitError);
          }
        }
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
  }, []);

  // Manual retry initialization
  const retryInitialization = useCallback(async () => {
    initializationAttempted.current = false;
    setModelLoadError(null);
    setIsModelLoaded(false);
    isShuttingDown.current = false; // Reset shutdown flag
    
    const initializeSystem = async () => {
      setIsInitializing(true);
      setInitializationStatus('Retrying initialization...');
      
      try {
        console.log("Retrying system initialization...");
        
        const modelResult = await detectionService.loadModel();
        
        if (modelResult.success) {
          setIsModelLoaded(true);
          setInitializationStatus('Detection model loaded successfully');
          console.log("Detection model loaded on retry:", modelResult.message);
        } else {
          throw new Error(modelResult.message || 'Failed to load detection model');
        }
        
        await performSingleHealthCheck();
        startStatsMonitoring();
        
        setInitializationStatus('System ready');
        console.log("System retry initialization completed successfully");
        
      } catch (error) {
        console.error("Error during retry initialization:", error);
        setModelLoadError(`Retry initialization failed: ${error.message}`);
        setIsModelLoaded(false);
        setInitializationStatus(`Retry failed: ${error.message}`);
      } finally {
        setIsInitializing(false);
      }
    };

    await initializeSystem();
  }, [performSingleHealthCheck]);

  // REMOVED: Continuous health monitoring - now only on-demand

  // FIXED: Only stats monitoring (much less frequent)
  const startStatsMonitoring = useCallback(() => {
    if (statsInterval.current) {
      clearInterval(statsInterval.current);
    }
    
    const updateGlobalStats = async () => {
      if (isShuttingDown.current) {
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
    // FIXED: Reduced frequency - only every 10 seconds instead of 5
    statsInterval.current = setInterval(updateGlobalStats, 10000);
  }, []);

  // Stop all monitoring
  const stopMonitoring = useCallback(() => {
    console.log("🛑 Stopping all monitoring...");
    isShuttingDown.current = true;
    
    if (healthCheckInterval.current) {
      clearInterval(healthCheckInterval.current);
      healthCheckInterval.current = null;
    }
    if (statsInterval.current) {
      clearInterval(statsInterval.current);
      statsInterval.current = null;
    }
  }, []);

  // FIXED: Enhanced cleanup with proper shutdown signaling
  useEffect(() => {
    const handleBeforeUnload = async (event) => {
      if (cleanupRef.current) return;
      cleanupRef.current = true;
      isShuttingDown.current = true;
      
      try {
        console.log("Performing cleanup...");
        
        // Stop all monitoring first
        stopMonitoring();
        
        // Stop all detection services
        if (isDetectionActive) {
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
  }, [isDetectionActive, stopMonitoring]);
  
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
  
  // FIXED: Enhanced detection start with health check
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
    
    if (!isModelLoaded) {
      alert("Detection model is not loaded. Please wait for initialization to complete or try refreshing the page.");
      return;
    }
    
    // FIXED: Perform health check before starting detection
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
    isShuttingDown.current = false; // Reset shutdown flag when starting
    setIsDetectionActive(true);
  }, [cameraId, targetLabel, isModelLoaded, systemHealth.overall, cameras, detectionOptions, performSingleHealthCheck]);
  
  // FIXED: Enhanced detection stop with post-shutdown health check
  const handleStopDetection = useCallback(async () => {
    console.log("Stopping detection...");
    isShuttingDown.current = true;
    setIsDetectionActive(false);
    
    // FIXED: Perform health check after shutdown to verify clean state
    setTimeout(async () => {
      console.log("🩺 Checking system health after detection stop...");
      isShuttingDown.current = false; // Allow health check after shutdown
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
        
        if (isDetectionActive) {
          alert("The camera currently in use is no longer available. Detection has been stopped.");
          setIsDetectionActive(false);
        }
      }
      
      console.log(`Successfully detected ${detectedCameras.length} cameras`);
    } catch (error) {
      console.error("Error detecting cameras:", error);
      alert(`Camera detection failed: ${error.message}`);
    } finally {
      setIsDetecting(false);
    }
  }, [selectedCameraId, isDetectionActive]);

  // Handle detection options changes
  const handleDetectionOptionsChange = useCallback((newOptions) => {
    setDetectionOptions(prev => ({
      ...prev,
      ...newOptions
    }));
    console.log("Detection options updated:", newOptions);
  }, []);

  // FIXED: Manual health check button handler
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

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Initialization Status */}
      {isInitializing && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {initializationStatus}
        </Alert>
      )}

      {/* System Status Alerts */}
      {modelLoadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {modelLoadError}
          <Box sx={{ mt: 1 }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={retryInitialization}
              disabled={isInitializing}
            >
              {isInitializing ? 'Retrying...' : 'Retry Initialization'}
            </Button>
          </Box>
          <br />
          <small>If the issue persists, try refreshing the page or contact support.</small>
        </Alert>
      )}
      
      {!isModelLoaded && !modelLoadError && !isInitializing && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Detection model not loaded. 
          <Button 
            variant="outlined" 
            size="small" 
            onClick={retryInitialization}
            sx={{ ml: 1 }}
          >
            Try Again
          </Button>
        </Alert>
      )}

      {!systemHealth.overall && isModelLoaded && !isInitializing && (
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
              disabled={isShuttingDown.current}
            >
              {isShuttingDown.current ? 'System Shutting Down...' : 'Check Health Now'}
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
              isDetectionActive={isDetectionActive}
              isModelLoaded={isModelLoaded && !isInitializing}
              systemHealth={systemHealth}
              detectionOptions={detectionOptions}
              onDetectionOptionsChange={handleDetectionOptionsChange}
            />
            
            <DetectionVideoFeed
              isDetectionActive={isDetectionActive}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              cameraId={cameraId}
              targetLabel={targetLabel}
              isModelLoaded={isModelLoaded && !isInitializing}
              setIsModelLoaded={setIsModelLoaded}
              detectionOptions={detectionOptions}
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
                {/* Initialization Status */}
                {isInitializing && (
                  <Box>
                    <Typography variant="subtitle2" color="textSecondary">
                      Initialization Status
                    </Typography>
                    <Chip
                      label="Initializing..."
                      color="warning"
                      size="small"
                      sx={{ mt: 0.5 }}
                      icon={<CircularProgress size={16} />}
                    />
                  </Box>
                )}

                {/* FIXED: System Health with last check time */}
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    System Health
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                    <Chip
                      label={
                        isInitializing ? "Initializing" : 
                        isShuttingDown.current ? "Shutting Down" :
                        systemHealth.overall ? "Healthy" : "Issues Detected"
                      }
                      color={
                        isInitializing ? "warning" :
                        isShuttingDown.current ? "info" :
                        systemHealth.overall ? "success" : "error"
                      }
                      size="small"
                    />
                    <Button
                      size="small"
                      variant="text"
                      onClick={handleManualHealthCheck}
                      disabled={isShuttingDown.current || isInitializing}
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