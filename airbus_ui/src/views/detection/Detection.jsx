// pages/AppDetection.jsx - Enhanced with adaptive system integration
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
  Button,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  Refresh, 
  Speed, 
  Computer, 
  Smartphone,
  AutoMode,
  Settings,
  Info
} from '@mui/icons-material';

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

// Performance modes
const PerformanceModes = {
  BASIC: 'basic',
  STANDARD: 'standard', 
  ENHANCED: 'enhanced',
  HIGH_PERFORMANCE: 'high_performance'
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

  // Adaptive system state
  const [systemProfile, setSystemProfile] = useState(null);
  const [currentPerformanceMode, setCurrentPerformanceMode] = useState('basic');
  const [currentStreamingType, setCurrentStreamingType] = useState('basic');
  const [autoModeEnabled, setAutoModeEnabled] = useState(true);
  const [systemCapabilities, setSystemCapabilities] = useState(null);
  const [isProfileRefreshing, setIsProfileRefreshing] = useState(false);

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
  const profileUpdateUnsubscribe = useRef(null);
  const healthCheckPerformed = useRef({
    initial: false,
    postShutdown: false
  });

  // Subscribe to detection service state changes
  useEffect(() => {
    const unsubscribe = detectionService.addStateChangeListener((newState, oldState) => {
      console.log(`🔄 Detection state changed: ${oldState} → ${newState}`);
      setDetectionState(newState);
      
      // Reset health check flags on state transitions
      if (newState === DetectionStates.INITIALIZING) {
        healthCheckPerformed.current.initial = false;
        healthCheckPerformed.current.postShutdown = false;
      } else if (newState === DetectionStates.READY && oldState === DetectionStates.SHUTTING_DOWN) {
        // Allow post-shutdown health check
        healthCheckPerformed.current.postShutdown = false;
      }
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

  // Subscribe to system profile updates
  useEffect(() => {
    const unsubscribe = detectionService.addProfileUpdateListener((profileData) => {
      console.log(`📊 System profile updated`, profileData);
      setSystemProfile(profileData.profile);
      setCurrentPerformanceMode(profileData.performanceMode);
      setCurrentStreamingType(profileData.streamingType);
      setSystemCapabilities(profileData.capabilities);
      setAutoModeEnabled(detectionService.autoModeEnabled);
    });
    
    profileUpdateUnsubscribe.current = unsubscribe;
    
    // Get initial profile
    const initialProfile = detectionService.getSystemProfile();
    const initialMode = detectionService.getCurrentPerformanceMode();
    const initialStreamingType = detectionService.getCurrentStreamingType();
    const initialCapabilities = detectionService.getSystemCapabilities();
    
    if (initialProfile) {
      setSystemProfile(initialProfile);
      setCurrentPerformanceMode(initialMode);
      setCurrentStreamingType(initialStreamingType);
      setSystemCapabilities(initialCapabilities);
      setAutoModeEnabled(detectionService.autoModeEnabled);
    }
    
    return () => {
      if (profileUpdateUnsubscribe.current) {
        profileUpdateUnsubscribe.current();
      }
    };
  }, []);

  // Initialize system on component mount
  useEffect(() => {
    const initializeSystem = async () => {
      if (initializationAttempted.current) return;
      initializationAttempted.current = true;

      try {
        console.log("🚀 Starting adaptive detection system initialization...");
        
        // Initialize detection processor (it will auto-select mode based on system)
        const initResult = await detectionService.ensureInitialized();
        
        if (initResult.success) {
          console.log("✅ Adaptive detection system initialized:", initResult.message);
          console.log(`📊 Selected mode: ${initResult.mode || detectionService.getCurrentStreamingType()}`);
          setInitializationError(null);
          
          // Perform initial health check right after initialization
          await performInitialHealthCheck();
          
          // Start stats monitoring
          startStatsMonitoring();
          
          console.log("✅ System initialization completed successfully");
        } else {
          throw new Error(initResult.message || 'Failed to initialize adaptive detection system');
        }
        
      } catch (error) {
        console.error("❌ Error initializing adaptive system:", error);
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

  // Watch for state transitions to trigger health checks
  useEffect(() => {
    const handleStateTransition = async () => {
      if (detectionState === DetectionStates.READY) {
        const serviceStatus = detectionService.getDetailedStatus();
        
        // If we just transitioned to READY after shutdown, perform post-shutdown health check
        if (!healthCheckPerformed.current.postShutdown && 
            !serviceStatus.hasPerformedPostShutdownCheck) {
          console.log("🩺 Triggering post-shutdown health check...");
          await performPostShutdownHealthCheck();
        }
        // If we just transitioned to READY after initialization, perform initial health check
        else if (!healthCheckPerformed.current.initial && 
                 !serviceStatus.hasPerformedInitialHealthCheck) {
          console.log("🩺 Triggering initial health check...");
          await performInitialHealthCheck();
        }
      }
    };

    handleStateTransition();
  }, [detectionState]);

  // Initial health check function
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckPerformed.current.initial) {
      console.log("⏭️ Initial health check already performed");
      return;
    }

    if (detectionState === DetectionStates.SHUTTING_DOWN) {
      console.log("⏭️ Skipping initial health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing initial health check...");
      const health = await detectionService.checkOptimizedHealth(true, false); // isInitialCheck = true
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.initial = true;
      
      console.log("✅ Initial health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Initial health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.initial = true;
    }
  }, [detectionState]);

  // Post-shutdown health check function
  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckPerformed.current.postShutdown) {
      console.log("⏭️ Post-shutdown health check already performed");
      return;
    }

    try {
      console.log("🩺 Performing post-shutdown health check...");
      const health = await detectionService.checkOptimizedHealth(false, true); // isPostShutdownCheck = true
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.postShutdown = true;
      
      console.log("✅ Post-shutdown health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Post-shutdown health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.postShutdown = true;
    }
  }, []);

  // Manual retry initialization
  const retryInitialization = useCallback(async () => {
    initializationAttempted.current = false;
    setInitializationError(null);
    healthCheckPerformed.current.initial = false;
    healthCheckPerformed.current.postShutdown = false;
    
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

  // Force refresh system profile
  const handleRefreshSystemProfile = useCallback(async () => {
    setIsProfileRefreshing(true);
    try {
      console.log("🔄 Force refreshing system profile...");
      const result = await detectionService.forceSystemProfileRefresh();
      
      if (result.success) {
        console.log("✅ System profile refreshed successfully");
        console.log(`📊 New mode: ${result.streamingType}, Performance: ${result.performanceMode}`);
      }
    } catch (error) {
      console.error("❌ Error refreshing system profile:", error);
      alert(`Failed to refresh system profile: ${error.message}`);
    } finally {
      setIsProfileRefreshing(false);
    }
  }, []);

  // Manual mode switching functions
  const handleSwitchToBasicMode = useCallback(async () => {
    if (detectionState === DetectionStates.RUNNING) {
      alert("Please stop detection before switching modes.");
      return;
    }

    try {
      const result = await detectionService.switchToBasicMode();
      console.log("✅ Switched to basic mode:", result);
    } catch (error) {
      console.error("❌ Error switching to basic mode:", error);
      alert(`Failed to switch to basic mode: ${error.message}`);
    }
  }, [detectionState]);

  const handleSwitchToOptimizedMode = useCallback(async () => {
    if (detectionState === DetectionStates.RUNNING) {
      alert("Please stop detection before switching modes.");
      return;
    }

    try {
      const result = await detectionService.switchToOptimizedMode();
      console.log("✅ Switched to optimized mode:", result);
    } catch (error) {
      console.error("❌ Error switching to optimized mode:", error);
      alert(`Failed to switch to optimized mode: ${error.message}`);
    }
  }, [detectionState]);

  const handleEnableAutoMode = useCallback(async () => {
    try {
      const result = await detectionService.enableAutoMode();
      console.log("✅ Auto mode enabled:", result);
    } catch (error) {
      console.error("❌ Error enabling auto mode:", error);
      alert(`Failed to enable auto mode: ${error.message}`);
    }
  }, []);

  // Run performance test
  const handleRunPerformanceTest = useCallback(async () => {
    if (detectionState === DetectionStates.RUNNING) {
      alert("Please stop detection before running performance test.");
      return;
    }

    try {
      console.log("🧪 Running performance test...");
      const result = await detectionService.runPerformanceTest(10); // 10 seconds test
      console.log("✅ Performance test completed:", result);
      
      // The profile will be updated automatically via the listener
      alert(`Performance test completed. New performance score: ${result.performance_score}/100`);
    } catch (error) {
      console.error("❌ Error running performance test:", error);
      alert(`Performance test failed: ${error.message}`);
    }
  }, [detectionState]);

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
    

    
    // Perform health check before starting detection if not done recently
    const timeSinceLastCheck = lastHealthCheck.current ? Date.now() - lastHealthCheck.current : Infinity;
    if (timeSinceLastCheck > 30000 || !systemHealth.overall) { // 30 seconds
      console.log("🩺 Checking system health before starting detection...");
      await performSingleHealthCheck();
    }
    
    if (!systemHealth.overall) {
      const proceed = window.confirm(
        `System health check indicates issues (${currentStreamingType} mode). Do you want to proceed anyway?`
      );
      if (!proceed) return;
    }
    
    // Validate camera still exists
    const cameraExists = cameras.some(cam => cam.id.toString() === cameraId.toString());
    if (!cameraExists) {
      alert("Selected camera is no longer available. Please detect cameras and select a new one.");
      return;
    }
    
    console.log(`Starting ${currentStreamingType} detection with options:`, detectionOptions);
  }, [cameraId, targetLabel, detectionState, systemHealth.overall, cameras, detectionOptions, currentStreamingType]);
  
  // Enhanced detection stop
  const handleStopDetection = useCallback(async () => {
    console.log(`Stopping ${currentStreamingType} detection...`);
    
    // The state will be managed by the service itself
    // Just trigger the stop, don't manage state here
    
    // Perform post-shutdown health check after shutdown to verify clean state
    setTimeout(async () => {
      console.log("🩺 Checking system health after detection stop...");
      await performPostShutdownHealthCheck();
    }, 3000); // Wait 3 seconds for shutdown to complete
  }, [performPostShutdownHealthCheck, currentStreamingType]);

  // Single health check function (for manual checks)
  const performSingleHealthCheck = useCallback(async () => {
    if (detectionState === DetectionStates.SHUTTING_DOWN) {
      console.log("⏭️ Skipping health check - system is shutting down");
      return;
    }

    try {
      console.log(`🩺 Performing manual health check (${currentStreamingType} mode)...`);
      const health = await detectionService.checkOptimizedHealth();
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      if (!health.overall) {
        console.warn("System health check failed:", health);
      }
      
      console.log("✅ Manual health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Manual health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
    }
  }, [detectionState, currentStreamingType]);

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
          message: 'Initializing adaptive detection system...',
          canOperate: false
        };
      case DetectionStates.READY:
        return {
          color: 'success',
          message: `System ready for ${currentStreamingType} detection`,
          canOperate: true
        };
      case DetectionStates.RUNNING:
        return {
          color: 'warning',
          message: `${currentStreamingType.toUpperCase()} detection active`,
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

  // Get mode display info
  const getModeDisplayInfo = () => {
    const isBasic = currentStreamingType === 'basic';
    return {
      icon: isBasic ? <Smartphone /> : <Computer />,
      color: isBasic ? 'warning' : 'success',
      description: isBasic 
        ? 'On-Demand Detection - Captures single frames when requested' 
        : 'Real-Time Detection - Continuous video stream analysis'
    };
  };

  const stateInfo = getStateInfo();
  const modeInfo = getModeDisplayInfo();

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Adaptive System Status Alert */}
      {detectionState === DetectionStates.INITIALIZING && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing adaptive detection system... Analyzing system capabilities.
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

      {/* System Mode Information */}
      {detectionState === DetectionStates.READY && systemProfile && (
        <Alert 
          severity="info" 
          sx={{ mb: 2 }}
          icon={modeInfo.icon}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                {currentStreamingType.toUpperCase()} MODE SELECTED
              </Typography>
              <Typography variant="body2">
                {modeInfo.description}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Performance Score: {systemProfile.performance_score}/100 | 
                CPU: {systemProfile.cpu_cores} cores | 
                RAM: {systemProfile.available_memory_gb}GB | 
                GPU: {systemProfile.gpu_available ? systemProfile.gpu_name : 'None'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Refresh System Profile">
                <IconButton 
                  size="small" 
                  onClick={handleRefreshSystemProfile}
                  disabled={isProfileRefreshing}
                >
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Run Performance Test">
                <IconButton 
                  size="small" 
                  onClick={handleRunPerformanceTest}
                  disabled={detectionState !== DetectionStates.READY}
                >
                  <Speed />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Alert>
      )}

      {/* System Status Alerts */}
      {initializationError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Adaptive system initialization failed: {initializationError}
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
          System health check indicates issues in {currentStreamingType} mode. Detection may not work optimally.
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
              // onStartDetection={handleStartDetection}
              // onStopDetection={handleStopDetection}
              // isDetectionActive={detectionState === DetectionStates.RUNNING}
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

                {/* Health Check Status */}
                <Divider />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Health Check Status
                  </Typography>
                  <Stack spacing={0.5} sx={{ mt: 1 }}>
                    <Typography variant="caption">
                      Initial: {healthCheckPerformed.current.initial ? '✅ Done' : '⏳ Pending'}
                    </Typography>
                    <Typography variant="caption">
                      Post-Shutdown: {healthCheckPerformed.current.postShutdown ? '✅ Done' : '⏳ Pending'}
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