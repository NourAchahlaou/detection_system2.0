// pages/AppDetection.jsx - Fixed to properly pass performance mode to child components
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
import DetectionVideoFeed from "./components/DetectionVideoFeed"; // FIXED: Use the modified DetectionVideoFeed
import { cameraService } from "../captureImage/CameraService";
import { detectionService } from "./detectionService"; // High-performance service
import { systemPerformanceService } from "./systemPerformanceService"; // System performance service

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
  const [isInitializing, setIsInitializing] = useState(false); // FIXED: Start as false
  const [initializationStatus, setInitializationStatus] = useState('');
  // Performance mode state
  const [performanceMode, setPerformanceMode] = useState(null);
  const [systemProfile, setSystemProfile] = useState(null);
  const [performanceModeLoading, setPerformanceModeLoading] = useState(true);
  const [performanceModeError, setPerformanceModeError] = useState(null);
  
  // System health - now adaptive based on mode
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

  // Smart health checking refs
  const healthCheckInterval = useRef(null);
  const statsInterval = useRef(null);
  const cleanupRef = useRef(false);
  const initializationAttempted = useRef(false);
  const lastHealthCheck = useRef(null);
  const isShuttingDown = useRef(false);

  // FIXED: Expose refresh function to window for child components
  useEffect(() => {
    window.refreshSystemPerformance = refreshPerformanceMode;
    return () => {
      delete window.refreshSystemPerformance;
    };
  }, []);

  // Determine performance mode first, then initialize appropriate services
  useEffect(() => {
    const determinePerformanceMode = async () => {
      setPerformanceModeLoading(true);
      setPerformanceModeError(null);
      
      try {
        console.log("🔍 Determining optimal performance mode...");
        
        const result = await systemPerformanceService.getSystemProfile();
        
        setPerformanceMode(result.mode);
        setSystemProfile(result.profile);
        
        if (!result.success) {
          setPerformanceModeError(result.error);
        }
        
        console.log(`✅ Performance mode determined: ${result.mode}`, result.profile);
        
        // Update detection options based on performance mode
        if (result.mode === 'basic') {
          setDetectionOptions(prev => ({
            ...prev,
            detectionFps: 0, // No continuous detection in basic mode
            streamQuality: Math.min(prev.streamQuality, 75), // Lower quality for basic mode
            enableAdaptiveQuality: false,
            enableFrameSkipping: false
          }));
        }
        
      } catch (error) {
        console.error("❌ Error determining performance mode:", error);
        setPerformanceMode('basic'); // Default to basic on error
        setPerformanceModeError(error.message);
        setModelLoadError(`Performance detection failed: ${error.message}`);
      } finally {
        setPerformanceModeLoading(false);
      }
    };

    determinePerformanceMode();
  }, []);

  // Initialize system based on performance mode
  useEffect(() => {
    const initializeSystem = async () => {
      if (initializationAttempted.current || !performanceMode || performanceModeLoading) return;
      initializationAttempted.current = true;

      setIsInitializing(true);
      setInitializationStatus(`Initializing ${performanceMode} performance detection...`);
      
      try {
        console.log(`Starting ${performanceMode} performance system initialization...`);
        
        if (performanceMode === 'high') {
          // Initialize high-performance detection service (existing logic)
          setInitializationStatus('Loading high-performance detection model...');
          const modelResult = await detectionService.loadModel();
          
          if (modelResult.success) {
            setIsModelLoaded(true);
            setInitializationStatus('High-performance detection model loaded successfully');
            console.log("High-performance detection model loaded:", modelResult.message);
          } else {
            throw new Error(modelResult.message || 'Failed to load high-performance detection model');
          }
          
          // Check high-performance system health
          setInitializationStatus('Checking high-performance system health...');
          await performSingleHealthCheck();
          
        } else {
          // For basic mode, we don't need to initialize the heavy detection service
          // The basic detection service will be initialized when needed
          setIsModelLoaded(true); // Basic mode is always "ready"
          setInitializationStatus('Basic detection mode ready');
          console.log("Basic detection mode initialized");
          
          // Check basic system health (camera availability, etc.)
          setInitializationStatus('Checking basic system health...');
          await performBasicHealthCheck();
        }
        
        // Start monitoring based on mode
        setInitializationStatus('Starting system monitoring...');
        startStatsMonitoring();
        
        setInitializationStatus(`${performanceMode === 'high' ? 'High-performance' : 'Basic'} system ready`);
        console.log(`${performanceMode} performance system initialization completed successfully`);
        
      } catch (error) {
        console.error(`Error initializing ${performanceMode} performance system:`, error);
        setModelLoadError(`${performanceMode === 'high' ? 'High-performance' : 'Basic'} system initialization failed: ${error.message}`);
        setIsModelLoaded(false);
        setInitializationStatus(`${performanceMode} initialization failed: ${error.message}`);
      } finally {
        setIsInitializing(false);
      }
    };

    initializeSystem();
    
    return () => {
      stopMonitoring();
    };
  }, [performanceMode, performanceModeLoading]);

  // Basic health check for basic mode
  const performBasicHealthCheck = useCallback(async () => {
    if (isShuttingDown.current) {
      console.log("⏭️ Skipping basic health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing basic system health check...");
      
      // Check if cameras are available
      const cameraHealth = await cameraService.getAllCameras();
      const hasAvailableCameras = cameraHealth && cameraHealth.length > 0;
      
      // For basic mode, we mainly need cameras to be available
      const health = {
        streaming: { 
          status: hasAvailableCameras ? 'healthy' : 'unhealthy',
          message: hasAvailableCameras ? 'Cameras available' : 'No cameras available'
        },
        detection: { 
          status: 'healthy', // Basic detection is always ready
          message: 'Basic detection ready'
        },
        overall: hasAvailableCameras
      };
      
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      console.log("✅ Basic health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Basic health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
    }
  }, []);

  // Enhanced health check that works for both modes
  const performSingleHealthCheck = useCallback(async () => {
    if (performanceMode === 'basic') {
      return await performBasicHealthCheck();
    }
    
    // Original high-performance health check logic
    if (isShuttingDown.current) {
      console.log("⏭️ Skipping health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing high-performance health check...");
      const health = await detectionService.checkOptimizedHealth();
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      if (!health.overall) {
        console.warn("High-performance system health check failed:", health);
        
        if (health.detection.status === 'unhealthy' && !isShuttingDown.current) {
          console.log("High-performance detection service unhealthy, attempting re-initialization...");
          try {
            await detectionService.ensureInitialized();
            const retryHealth = await detectionService.checkOptimizedHealth();
            setSystemHealth(retryHealth);
            lastHealthCheck.current = Date.now();
          } catch (reinitError) {
            console.error("Re-initialization failed:", reinitError);
          }
        }
      }
      
      console.log("✅ High-performance health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("High-performance health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
    }
  }, [performanceMode, performBasicHealthCheck]);

  // FIXED: Refresh performance mode function
  const refreshPerformanceMode = useCallback(async () => {
    setPerformanceModeLoading(true);
    setPerformanceModeError(null);
    
    try {
      console.log("🔄 Refreshing performance mode analysis...");
      const result = await systemPerformanceService.getSystemProfile(true); // Force refresh
      
      setPerformanceMode(result.mode);
      setSystemProfile(result.profile);
      
      if (!result.success) {
        setPerformanceModeError(result.error);
      }
      
      // Reset initialization to use new mode
      initializationAttempted.current = false;
      setIsInitializing(false);
      setIsModelLoaded(false);
      
      console.log(`✅ Performance mode refreshed: ${result.mode}`);
    } catch (error) {
      console.error("❌ Error refreshing performance mode:", error);
      setPerformanceModeError(error.message);
    } finally {
      setPerformanceModeLoading(false);
    }
  }, []);

  // Manual retry initialization
  const retryInitialization = useCallback(async () => {
    initializationAttempted.current = false;
    setModelLoadError(null);
    setIsModelLoaded(false);
    isShuttingDown.current = false;
    
    // Re-trigger initialization based on current performance mode
    const initializeSystem = async () => {
      setIsInitializing(true);
      setInitializationStatus('Retrying initialization...');
      
      try {
        console.log(`Retrying ${performanceMode} performance system initialization...`);
        
        if (performanceMode === 'high') {
          const modelResult = await detectionService.loadModel();
          
          if (modelResult.success) {
            setIsModelLoaded(true);
            setInitializationStatus('High-performance detection model loaded successfully');
            console.log("High-performance detection model loaded on retry:", modelResult.message);
          } else {
            throw new Error(modelResult.message || 'Failed to load high-performance detection model');
          }
          
          await performSingleHealthCheck();
        } else {
          setIsModelLoaded(true);
          setInitializationStatus('Basic detection mode ready');
          await performBasicHealthCheck();
        }
        
        startStatsMonitoring();
        setInitializationStatus(`${performanceMode === 'high' ? 'High-performance' : 'Basic'} system ready`);
        console.log(`${performanceMode} performance system retry initialization completed successfully`);
        
      } catch (error) {
        console.error(`Error during ${performanceMode} retry initialization:`, error);
        setModelLoadError(`${performanceMode === 'high' ? 'High-performance' : 'Basic'} retry initialization failed: ${error.message}`);
        setIsModelLoaded(false);
        setInitializationStatus(`${performanceMode} retry failed: ${error.message}`);
      } finally {
        setIsInitializing(false);
      }
    };

    await initializeSystem();
  }, [performanceMode, performSingleHealthCheck, performBasicHealthCheck]);

  // Enhanced stats monitoring that works for both modes
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
        if (performanceMode === 'high') {
          // Use existing high-performance stats
          const stats = await detectionService.getAllStreamingStats();
          
          setGlobalStats({
            totalStreams: stats.active_streams || 0,
            avgProcessingTime: stats.avg_processing_time_ms || 0,
            totalDetections: stats.total_detections || 0,
            systemLoad: stats.system_load_percent || 0,
            memoryUsage: stats.memory_usage_mb || 0
          });
        } else {
          // For basic mode, we have simpler stats
          setGlobalStats({
            totalStreams: isDetectionActive ? 1 : 0,
            avgProcessingTime: 0, // Will be updated when detection happens
            totalDetections: 0, // Will be updated by the adaptive component
            systemLoad: 0, // Basic mode has minimal load
            memoryUsage: 0 // Basic mode has minimal memory usage
          });
        }
      } catch (error) {
        console.debug("Error fetching global stats:", error);
      }
    };

    updateGlobalStats();
    // Reduced frequency for basic mode
    const interval = performanceMode === 'high' ? 10000 : 15000;
    statsInterval.current = setInterval(updateGlobalStats, interval);
  }, [performanceMode, isDetectionActive]);

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

  // Enhanced cleanup for both modes
  useEffect(() => {
    const handleBeforeUnload = async (event) => {
      if (cleanupRef.current) return;
      cleanupRef.current = true;
      isShuttingDown.current = true;
      
      try {
        console.log("Performing cleanup...");
        
        stopMonitoring();
        
        if (isDetectionActive) {
          if (performanceMode === 'high') {
            await detectionService.stopAllStreams();
          }
          // Basic mode cleanup is handled by the adaptive component
        }
        
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
  }, [isDetectionActive, performanceMode, stopMonitoring]);
  
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
  
  // Enhanced detection start that works for both modes
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
      alert("Detection system is not loaded. Please wait for initialization to complete or try refreshing the page.");
      return;
    }
    
    // Perform health check before starting detection
    console.log("🩺 Checking system health before starting detection...");
    await performSingleHealthCheck();
    
    if (!systemHealth.overall) {
      const proceed = window.confirm(
        `System health check indicates issues. Do you want to proceed anyway? (Mode: ${performanceMode})`
      );
      if (!proceed) return;
    }
    
    // Validate camera still exists
    const cameraExists = cameras.some(cam => cam.id.toString() === cameraId.toString());
    if (!cameraExists) {
      alert("Selected camera is no longer available. Please detect cameras and select a new one.");
      return;
    }
    
    console.log(`Starting ${performanceMode} performance detection with options:`, detectionOptions);
    isShuttingDown.current = false;
    setIsDetectionActive(true);
  }, [cameraId, targetLabel, isModelLoaded, systemHealth.overall, cameras, detectionOptions, performanceMode, performSingleHealthCheck]);
  
  // Enhanced detection stop that works for both modes
  const handleStopDetection = useCallback(async () => {
    console.log(`Stopping ${performanceMode} performance detection...`);
    isShuttingDown.current = true;
    setIsDetectionActive(false);
    
    setTimeout(async () => {
      console.log(`🩺 Checking ${performanceMode} system health after detection stop...`);
      isShuttingDown.current = false;
      await performSingleHealthCheck();
    }, 2000);
  }, [performanceMode, performSingleHealthCheck]);

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

  // Manual health check button handler
  const handleManualHealthCheck = useCallback(async () => {
    console.log(`🩺 Manual ${performanceMode} health check requested...`);
    await performSingleHealthCheck();
  }, [performanceMode, performSingleHealthCheck]);

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

  // Show loading while determining performance mode
  if (performanceModeLoading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 3 }}>
        <CircularProgress size={40} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Analyzing System Performance...
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Determining optimal detection mode for your hardware
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Performance Mode Info */}
      <Alert 
        severity={performanceModeError ? "warning" : "info"} 
        sx={{ mb: 2 }}
        icon={performanceMode === 'high' ? '⚡' : '🎯'}
        action={
          <Button 
            size="small" 
            onClick={refreshPerformanceMode}
            disabled={performanceModeLoading}
          >
            Refresh
          </Button>
        }
      >
        <Typography variant="body2">
          <strong>{performanceMode === 'high' ? 'High Performance Mode' : 'Basic Detection Mode'}</strong> - 
          {performanceMode === 'high' 
            ? ' Real-time detection with continuous processing' 
            : ' On-demand detection with manual triggers'
          }
        </Typography>
        {systemProfile && (
          <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
            CPU: {systemProfile.cpu_cores} cores | RAM: {systemProfile.total_memory_gb}GB | 
            GPU: {systemProfile.gpu_available ? 'Available' : 'Not Available'} | 
            Score: {systemProfile.performance_score}
          </Typography>
        )}
        {performanceModeError && (
          <Typography variant="caption" color="error" display="block" sx={{ mt: 0.5 }}>
            Warning: {performanceModeError}
          </Typography>
        )}
      </Alert>

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
      
      {!isModelLoaded && !modelLoadError && !isInitializing && performanceMode && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Detection system not loaded. 
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
              performanceMode={performanceMode}
            />
            
            {/* FIXED: Pass performance mode and system profile as props */}
            <DetectionVideoFeed
              isDetectionActive={isDetectionActive}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              cameraId={cameraId}
              targetLabel={targetLabel}
              isModelLoaded={isModelLoaded && !isInitializing}
              onModelLoadedChange={setIsModelLoaded}
              detectionOptions={detectionOptions}
              performanceMode={performanceMode} // Pass performance mode
              systemProfile={systemProfile} // Pass system profile
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
                {/* Performance Mode Display */}
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Detection Mode
                  </Typography>
                  <Chip
                    label={performanceMode === 'high' ? 'High Performance' : 'Basic Mode'}
                    color={performanceMode === 'high' ? 'success' : 'info'}
                    size="small"
                    icon={performanceMode === 'high' ? '⚡' : '🎯'}
                    sx={{ mt: 0.5 }}
                  />
                </Box>

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

                {/* System Health */}
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

                {performanceMode === 'high' && (
                  <>
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
                  </>
                )}

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