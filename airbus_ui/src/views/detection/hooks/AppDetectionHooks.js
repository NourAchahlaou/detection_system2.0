// hooks/AppDetectionHooks.js - FIXED: Updated with proper lot workflow parameter support
import { useState, useEffect, useCallback, useRef } from "react";
import { cameraService } from "../../captureImage/CameraService";
import { detectionService } from "../service/DetectionService";
import api from "../../../utils/UseAxios";

// Detection states from service
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// FIXED: Custom hook for lot management with proper lot workflow parameter support
export const useLotManagement = (selectedLotId = null, lotWorkflowActive = false) => {
  const [existingLots, setExistingLots] = useState([]);
  const [currentLot, setCurrentLot] = useState(null);
  const [lotOperationInProgress, setLotOperationInProgress] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  // Fetch control refs to prevent infinite loops
  const fetchControlRef = useRef({
    isFetching: false,
    lastFetchTime: 0,
    fetchPromise: null,
    abortController: null
  });

  // Show snackbar message
  const showSnackbar = useCallback((message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  // Optimized fetch existing lots - FIXED to prevent infinite loops
  const fetchExistingLots = useCallback(async (forceRefresh = false) => {
    const CACHE_DURATION = 30000; // 30 seconds
    const now = Date.now();
    
    // Check if we should skip this fetch
    if (!forceRefresh) {
      if (fetchControlRef.current.isFetching) {
        console.log("📋 Fetch already in progress, returning existing promise");
        return fetchControlRef.current.fetchPromise;
      }
      
      if ((now - fetchControlRef.current.lastFetchTime) < CACHE_DURATION) {
        console.log("📋 Using cached lots data (recent fetch)");
        return;
      }
    }

    // Cancel any previous request
    if (fetchControlRef.current.abortController) {
      fetchControlRef.current.abortController.abort();
    }

    // Create new abort controller
    fetchControlRef.current.abortController = new AbortController();
    fetchControlRef.current.isFetching = true;
    
    // Create fetch promise
    const fetchPromise = (async () => {
      try {
        console.log("📋 Fetching existing detection lots...");
        
        const response = await api.get('/api/detection/basic/lots?limit=50', {
          signal: fetchControlRef.current.abortController.signal
        });
        
        if (response.data.success) {
          const lots = response.data.lots || [];
          setExistingLots(lots);
          fetchControlRef.current.lastFetchTime = now;
          console.log(`✅ Loaded ${lots.length} existing lots`);
          return lots;
        } else {
          throw new Error(response.data.message || 'Failed to fetch lots');
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log("📋 Fetch request was aborted");
          return;
        }
        
        console.error("❌ Error fetching existing lots:", error);
        showSnackbar("Failed to load existing lots", "warning");
        return existingLots; // Return current state on error
      } finally {
        fetchControlRef.current.isFetching = false;
        fetchControlRef.current.fetchPromise = null;
        fetchControlRef.current.abortController = null;
      }
    })();

    fetchControlRef.current.fetchPromise = fetchPromise;
    return fetchPromise;
  }, []); // No dependencies to prevent loops

  // Force refresh lots - for manual refresh only
  const refreshLots = useCallback(async () => {
    console.log("🔄 Force refreshing lots...");
    return fetchExistingLots(true);
  }, [fetchExistingLots]);

  // Create lot only (without detection) - NEW FUNCTION
  const handleCreateLotOnly = useCallback(async (lotData) => {
    if (!lotData.lotName || !lotData.expectedPieceId || !lotData.expectedPieceNumber) {
      showSnackbar("Lot name, piece ID, and piece number are required", "error");
      return;
    }

    setLotOperationInProgress(true);

    try {
      console.log("📦 Creating lot only...", lotData);

      const response = await api.post('/api/detection/basic/lots', {
        lot_name: lotData.lotName,
        expected_piece_id: lotData.expectedPieceId,
        expected_piece_number: lotData.expectedPieceNumber
      });

      if (response.data.success) {
        const lotCreated = response.data.lot;
        setCurrentLot(lotCreated);

        showSnackbar(
          `Lot "${lotCreated.lot_name}" created successfully!`,
          'success'
        );

        console.log("✅ Lot created:", lotCreated);

        // Refresh lots after successful creation
        await refreshLots();
        return lotCreated;
      } else {
        throw new Error(response.data.message || 'Failed to create lot');
      }
    } catch (error) {
      console.error("❌ Error creating lot:", error);
      showSnackbar(
        `Failed to create lot: ${error.response?.data?.detail || error.message}`,
        "error"
      );
      throw error;
    } finally {
      setLotOperationInProgress(false);
    }
  }, [showSnackbar, refreshLots]);

  // FIXED: Create new lot and perform detection with lot workflow parameters
  const handleCreateLotAndDetect = useCallback(async (lotData, cameraId, targetLabel, detectionOptions) => {
    if (!cameraId || !targetLabel) {
      showSnackbar("Camera ID and target label are required", "error");
      return;
    }

    if (!lotData.lotName || !lotData.expectedPieceId || !lotData.expectedPieceNumber) {
      showSnackbar("Lot name, piece ID, and piece number are required", "error");
      return;
    }

    setLotOperationInProgress(true);

    try {
      console.log("🚀 Creating lot and performing detection...", {
        lotData,
        selectedLotId,
        lotWorkflowActive
      });

      // Ensure all required fields are present and properly typed
      const requestData = {
        lot_name: lotData.lotName.trim(),
        expected_piece_id: parseInt(lotData.expectedPieceId),
        expected_piece_number: parseInt(lotData.expectedPieceNumber),
        target_label: targetLabel,
        quality: detectionOptions.quality || 85,
        // Add workflow context if available
        workflow_active: lotWorkflowActive,
        parent_lot_id: selectedLotId || null,
        // Add any additional detection options
        ...(detectionOptions.priority && { priority: detectionOptions.priority }),
        ...(detectionOptions.enableAdaptiveQuality && { adaptive_quality: detectionOptions.enableAdaptiveQuality })
      };

      console.log("📤 Sending request with data:", requestData);

      const response = await api.post(
        `/api/detection/basic/detect/${cameraId}/with-lot-creation`,
        requestData,
        {
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data.success) {
        const lotCreated = response.data.lot_created;
        const detectionResult = response.data.detection_result;

        setCurrentLot(lotCreated);

        const successMessage = `Lot "${lotCreated.lot_name}" created and detection completed! ${
          detectionResult.detected_target ? '🎯 Target detected!' : 'Target not found'
        }`;

        showSnackbar(successMessage, detectionResult.detected_target ? 'success' : 'info');

        console.log("✅ Lot created and detection completed:", {
          lot: lotCreated,
          detection: detectionResult,
          workflowContext: { selectedLotId, lotWorkflowActive }
        });

        // Refresh lots after successful creation
        await refreshLots();
        
        return { 
          lotCreated, 
          detectionResult,
          success: true,
          workflowContext: { selectedLotId, lotWorkflowActive }
        };
      } else {
        throw new Error(response.data.message || 'Operation failed');
      }
    } catch (error) {
      console.error("❌ Error creating lot and detecting:", error);
      
      let errorMessage = "Failed to create lot and detect";
      
      if (error.response?.data?.detail) {
        errorMessage += `: ${error.response.data.detail}`;
      } else if (error.response?.data?.message) {
        errorMessage += `: ${error.response.data.message}`;
      } else if (error.message) {
        errorMessage += `: ${error.message}`;
      }

      showSnackbar(errorMessage, "error");
      
      return {
        success: false,
        error: errorMessage
      };
    } finally {
      setLotOperationInProgress(false);
    }
  }, [showSnackbar, refreshLots, selectedLotId, lotWorkflowActive]);

  // FIXED: Perform detection with existing lot with lot workflow parameters
  const handleDetectWithExistingLot = useCallback(async (lotData, cameraId, targetLabel, detectionOptions) => {
    if (!cameraId || !targetLabel) {
      showSnackbar("Camera ID and target label are required", "error");
      return;
    }

    if (!lotData.lotId && !lotData.selectedLotId) {
      showSnackbar("Lot ID is required", "error");
      return;
    }

    setLotOperationInProgress(true);

    try {
      const lotId = lotData.lotId || lotData.selectedLotId;
      console.log("🎯 Performing detection with existing lot...", { 
        lotId, 
        targetLabel,
        selectedLotId,
        lotWorkflowActive
      });

      const requestData = {
        lot_id: parseInt(lotId),
        target_label: targetLabel,
        quality: detectionOptions.quality || 85,
        auto_complete_on_match: true,
        // Add workflow context
        workflow_active: lotWorkflowActive,
        selected_lot_id: selectedLotId || null,
        // Add any additional detection options
        ...(detectionOptions.priority && { priority: detectionOptions.priority }),
        ...(detectionOptions.enableAdaptiveQuality && { adaptive_quality: detectionOptions.enableAdaptiveQuality })
      };

      const response = await api.post(
        `/api/detection/basic/detect/${cameraId}/with-auto-correction`,
        requestData,
        {
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data.success) {
        const detectionResult = response.data.detection_result;
        const correctionAction = response.data.correction_action;
        const lotCompleted = response.data.lot_completed;

        // Update current lot info
        if (lotCompleted) {
          const updatedLot = existingLots.find(lot => lot.lot_id === parseInt(lotId));
          if (updatedLot) {
            setCurrentLot({ 
              ...updatedLot, 
              is_target_match: true, 
              completed_at: new Date().toISOString() 
            });
          }
        }

        let message = `Detection completed! ${detectionResult.detected_target ? '🎯 Target detected!' : 'Target not found'}`;
        if (lotCompleted) {
          message += " Lot marked as complete!";
        } else if (correctionAction === 'needs_correction') {
          message += " Lot still needs correction.";
        }

        showSnackbar(message, detectionResult.detected_target ? 'success' : 'info');

        console.log("✅ Detection with existing lot completed:", {
          detection: detectionResult,
          correctionAction,
          lotCompleted,
          workflowContext: { selectedLotId, lotWorkflowActive }
        });

        // Refresh lots after detection
        await refreshLots();
        
        return { 
          detectionResult, 
          correctionAction, 
          lotCompleted,
          success: true,
          workflowContext: { selectedLotId, lotWorkflowActive }
        };
      } else {
        throw new Error(response.data.message || 'Detection operation failed');
      }
    } catch (error) {
      console.error("❌ Error detecting with existing lot:", error);
      
      let errorMessage = "Detection with lot failed";
      
      if (error.response?.data?.detail) {
        errorMessage += `: ${error.response.data.detail}`;
      } else if (error.response?.data?.message) {
        errorMessage += `: ${error.response.data.message}`;
      } else if (error.message) {
        errorMessage += `: ${error.message}`;
      }

      showSnackbar(errorMessage, "error");
      
      return {
        success: false,
        error: errorMessage
      };
    } finally {
      setLotOperationInProgress(false);
    }
  }, [existingLots, showSnackbar, refreshLots, selectedLotId, lotWorkflowActive]);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (fetchControlRef.current.abortController) {
      fetchControlRef.current.abortController.abort();
    }
    fetchControlRef.current.isFetching = false;
    fetchControlRef.current.fetchPromise = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    // State
    existingLots,
    setExistingLots,
    currentLot,
    setCurrentLot,
    lotOperationInProgress,
    setLotOperationInProgress,
    snackbar,
    setSnackbar,
    
    // Functions
    showSnackbar,
    fetchExistingLots, // For initial load only
    refreshLots, // For manual refresh
    handleCreateLotOnly, // NEW: Create lot without detection
    handleCreateLotAndDetect, // Create lot and perform detection
    handleDetectWithExistingLot, // Use existing lot for detection
    cleanup // For manual cleanup if needed
  };
};

// Custom hook for detection system management
export const useDetectionSystem = () => {
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

  // Basic mode specific state
  const [isStreamFrozen, setIsStreamFrozen] = useState(false);
  const [onDemandDetecting, setOnDemandDetecting] = useState(false);
  const [lastDetectionResult, setLastDetectionResult] = useState(null);
  const [detectionInProgress, setDetectionInProgress] = useState(false);

  // Performance state
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
  const freezeListenerUnsubscribe = useRef(null);
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
        healthCheckPerformed.current.postShutdown = false;
      }
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
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

  // Subscribe to freeze/unfreeze events for basic mode
  useEffect(() => {
    if (currentStreamingType !== 'basic') return;
    
    const unsubscribe = detectionService.addFreezeListener((freezeEvent) => {
      console.log(`🧊 Freeze event:`, freezeEvent);
      setIsStreamFrozen(freezeEvent.status === 'frozen');
    });
    
    freezeListenerUnsubscribe.current = unsubscribe;
    
    return () => {
      if (freezeListenerUnsubscribe.current) {
        freezeListenerUnsubscribe.current();
      }
    };
  }, [currentStreamingType]);

  // Health check functions
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
      const health = await detectionService.checkOptimizedHealth(true, false);
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

  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckPerformed.current.postShutdown) {
      console.log("⏭️ Post-shutdown health check already performed");
      return;
    }

    try {
      console.log("🩺 Performing post-shutdown health check...");
      const health = await detectionService.checkOptimizedHealth(false, true);
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

  // Stats monitoring
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

  const stopMonitoring = useCallback(() => {
    console.log("🛑 Stopping all monitoring...");
    
    if (statsInterval.current) {
      clearInterval(statsInterval.current);
      statsInterval.current = null;
    }
  }, []);

  // Helper functions
  const getHealthCheckAge = () => {
    if (!lastHealthCheck.current) return 'Never';
    const ageMs = Date.now() - lastHealthCheck.current;
    const ageSeconds = Math.floor(ageMs / 1000);
    if (ageSeconds < 60) return `${ageSeconds}s ago`;
    const ageMinutes = Math.floor(ageSeconds / 60);
    return `${ageMinutes}m ago`;
  };

  return {
    // State
    detectionState,
    initializationError,
    setInitializationError,
    systemHealth,
    systemProfile,
    currentPerformanceMode,
    currentStreamingType,
    autoModeEnabled,
    systemCapabilities,
    isProfileRefreshing,
    setIsProfileRefreshing,
    isStreamFrozen,
    setIsStreamFrozen,
    onDemandDetecting,
    setOnDemandDetecting,
    lastDetectionResult,
    setLastDetectionResult,
    detectionInProgress,
    setDetectionInProgress,
    globalStats,
    
    // Refs
    initializationAttempted,
    healthCheckPerformed,
    lastHealthCheck,
    cleanupRef,
    
    // Functions
    performInitialHealthCheck,
    performPostShutdownHealthCheck,
    performSingleHealthCheck,
    startStatsMonitoring,
    stopMonitoring,
    getHealthCheckAge,
    
    // Constants
    DetectionStates
  };
};

// Custom hook for camera management
export const useCameraManagement = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);

  // Fetch available cameras
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

  // Enhanced camera detection
  const handleDetectCameras = useCallback(async (showSnackbar) => {
    setIsDetecting(true);
    try {
      console.log("Detecting available cameras...");
      const detectedCameras = await cameraService.detectCameras();
      setCameras(detectedCameras);
      
      if (selectedCameraId && !detectedCameras.some(cam => cam.id.toString() === selectedCameraId.toString())) {
        console.log("Previously selected camera no longer available, resetting selection");
        setSelectedCameraId('');
        setCameraId('');
        
        if (showSnackbar) {
          showSnackbar("The camera currently in use is no longer available. Detection has been stopped.", "warning");
        }
      }
      
      console.log(`Successfully detected ${detectedCameras.length} cameras`);
    } catch (error) {
      console.error("Error detecting cameras:", error);
      if (showSnackbar) {
        showSnackbar(`Camera detection failed: ${error.message}`, "error");
      }
    } finally {
      setIsDetecting(false);
    }
  }, [selectedCameraId]);

  return {
    cameras,
    setCameras,
    selectedCameraId,
    setSelectedCameraId,
    cameraId,
    setCameraId,
    isDetecting,
    handleDetectCameras
  };
};