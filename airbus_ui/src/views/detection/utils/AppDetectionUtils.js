// utils/AppDetectionUtils.js - FIXED: Proper lot handling for detection
import { detectionService } from "../service/DetectionService";
import api from "../../../utils/UseAxios";

// Detection states from service
export const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// System initialization functions
export const initializeDetectionSystem = async (
  initializationAttempted,
  setInitializationError,
  performInitialHealthCheck,
  startStatsMonitoring
) => {
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

// Retry initialization function
export const createRetryInitialization = (
  initializationAttempted,
  setInitializationError,
  healthCheckPerformed
) => {
  return async () => {
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
  };
};

// System control functions
export const createSystemControlFunctions = (
  detectionState,
  showSnackbar,
  currentStreamingType
) => {
  // Force refresh system profile
  const handleRefreshSystemProfile = async (setIsProfileRefreshing) => {
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
      showSnackbar(`Failed to refresh system profile: ${error.message}`, "error");
    } finally {
      setIsProfileRefreshing(false);
    }
  };

  // Manual mode switching functions
  const handleSwitchToBasicMode = async () => {
    if (detectionState === DetectionStates.RUNNING) {
      showSnackbar("Please stop detection before switching modes", "warning");
      return;
    }

    try {
      const result = await detectionService.switchToBasicMode();
      console.log("✅ Switched to basic mode:", result);
    } catch (error) {
      console.error("❌ Error switching to basic mode:", error);
      showSnackbar(`Failed to switch to basic mode: ${error.message}`, "error");
    }
  };

  const handleSwitchToOptimizedMode = async () => {
    if (detectionState === DetectionStates.RUNNING) {
      showSnackbar("Please stop detection before switching modes", "warning");
      return;
    }

    try {
      const result = await detectionService.switchToOptimizedMode();
      console.log("✅ Switched to optimized mode:", result);
    } catch (error) {
      console.error("❌ Error switching to optimized mode:", error);
      showSnackbar(`Failed to switch to optimized mode: ${error.message}`, "error");
    }
  };

  const handleEnableAutoMode = async () => {
    try {
      const result = await detectionService.enableAutoMode();
      console.log("✅ Auto mode enabled:", result);
    } catch (error) {
      console.error("❌ Error enabling auto mode:", error);
      showSnackbar(`Failed to enable auto mode: ${error.message}`, "error");
    }
  };

  // Run performance test
  const handleRunPerformanceTest = async () => {
    if (detectionState === DetectionStates.RUNNING) {
      showSnackbar("Please stop detection before running performance test", "warning");
      return;
    }

    try {
      console.log("🧪 Running performance test...");
      const result = await detectionService.runPerformanceTest(10);
      console.log("✅ Performance test completed:", result);
      
      showSnackbar(`Performance test completed. New performance score: ${result.performance_score}/100`, "success");
    } catch (error) {
      console.error("❌ Error running performance test:", error);
      showSnackbar(`Performance test failed: ${error.message}`, "error");
    }
  };

  return {
    handleRefreshSystemProfile,
    handleSwitchToBasicMode,
    handleSwitchToOptimizedMode,
    handleEnableAutoMode,
    handleRunPerformanceTest
  };
};

// FIXED: Basic mode detection functions with proper lot handling
export const createBasicModeHandlers = (
  cameraId,
  targetLabel,
  currentLot,
  detectionInProgress,
  detectionOptions,
  showSnackbar,
  setDetectionInProgress,
  setOnDemandDetecting,
  setLastDetectionResult,
  setIsStreamFrozen,
  setCurrentLot,
  fetchExistingLots,
  // NEW: Additional parameters for lot workflow support
  selectedLotId = null,
  lotWorkflowActive = false
) => {
// FIXED: Enhanced on-demand detection with proper lot selection handling
const handleOnDemandDetection = async (options = {}) => {
  if (!cameraId || !targetLabel) {
    showSnackbar("Camera ID and target label are required for detection", "error");
    return;
  }

  if (detectionInProgress) {
    console.log("🚫 Detection already in progress, skipping...");
    return;
  }

  // FIXED: More intelligent lot checking for lot workflow
  let effectiveLot = currentLot;
  let effectiveLotId = null;
  
  // If we're in lot workflow mode and have a selected lot ID, use that
  if (lotWorkflowActive && selectedLotId) {
    effectiveLotId = selectedLotId;
    
    // If we don't have currentLot loaded yet, that's OK - we'll use the selectedLotId
    if (!effectiveLot) {
      console.log(`📋 Lot workflow active with selected lot ${selectedLotId}, currentLot will be loaded during detection`);
    } else if (effectiveLot.lot_id !== selectedLotId) {
      console.log(`📋 CurrentLot (${effectiveLot.lot_id}) doesn't match selectedLotId (${selectedLotId}), using selectedLotId`);
    }
  } else if (effectiveLot) {
    // We have a current lot, use its ID
    effectiveLotId = effectiveLot.lot_id;
  }
  
  // Check if we have either a lot ID or current lot for basic mode
  if (effectiveLotId || effectiveLot) {
    setDetectionInProgress(true);
    setOnDemandDetecting(true);

    try {
      const lotId = effectiveLotId || effectiveLot.lot_id;
      
      console.log(`🎯 Performing auto-correction detection for lot ${lotId}`);
      
      const response = await api.post(
        `/api/detection/basic/detect/${cameraId}/with-auto-correction`,
        {
          lot_id: lotId,
          target_label: targetLabel,
          quality: options.quality || detectionOptions.quality,
          auto_complete_on_match: true,
          // Add workflow context
          workflow_active: lotWorkflowActive,
          selected_lot_id: selectedLotId || null
        }
      );

      if (response.data.success) {
        const detectionResult = response.data.detection_result;
        const lotCompleted = response.data.lot_completed;

        setLastDetectionResult(detectionResult);
        setIsStreamFrozen(detectionResult.stream_frozen);

        if (lotCompleted) {
          // Update current lot to reflect completion
          if (effectiveLot) {
            setCurrentLot(prev => ({ ...prev, is_target_match: true, completed_at: new Date() }));
          }
          showSnackbar("🎯 Target detected! Lot completed successfully!", "success");
        } else {
          showSnackbar(
            detectionResult.detected_target 
              ? "Target detected but lot needs verification" 
              : "Target not found, try again", 
            "info"
          );
        }

        // Refresh lots list
        if (fetchExistingLots) {
          await fetchExistingLots();
        }
      }
    } catch (error) {
      console.error("❌ Auto-correction detection failed:", error);
      showSnackbar(`Detection failed: ${error.response?.data?.detail || error.message}`, "error");
    } finally {
      setOnDemandDetecting(false);
      setDetectionInProgress(false);
    }
  } else {
    // FIXED: More informative message based on context
    if (lotWorkflowActive) {
      showSnackbar("Lot workflow is active but no lot is selected. Please check the URL parameters or reload the page.", "warning");
    } else {
      showSnackbar("Please create or select a lot before performing detection", "warning");
    }
  }
};
  // Basic mode: Freeze stream
  const handleFreezeStream = async () => {
    if (!cameraId) return;

    try {
      console.log(`🧊 Freezing stream for camera ${cameraId}`);
      await detectionService.freezeStream(cameraId);
      console.log(`✅ Stream frozen for camera ${cameraId}`);
    } catch (error) {
      console.error("❌ Error freezing stream:", error);
      showSnackbar(`Failed to freeze stream: ${error.message}`, "error");
    }
  };

  // Basic mode: Unfreeze stream
  const handleUnfreezeStream = async () => {
    if (!cameraId) return;

    try {
      console.log(`🔥 Unfreezing stream for camera ${cameraId}`);
      await detectionService.unfreezeStream(cameraId);
      console.log(`✅ Stream unfrozen for camera ${cameraId}`);
    } catch (error) {
      console.error("❌ Error unfreezing stream:", error);
      showSnackbar(`Failed to unfreeze stream: ${error.message}`, "error");
    }
  };

  return {
    handleOnDemandDetection,
    handleFreezeStream,
    handleUnfreezeStream
  };
};

// FIXED: Detection control functions with proper lot validation
export const createDetectionControlHandlers = (
  cameraId,
  targetLabel,
  currentStreamingType,
  systemHealth,
  cameras,
  detectionOptions,
  showSnackbar,
  performSingleHealthCheck,
  lastHealthCheck,
  setIsStreamFrozen,
  setLastDetectionResult,
  setOnDemandDetecting,
  setDetectionInProgress,
  setCurrentLot,
  performPostShutdownHealthCheck,
  // NEW: Lot workflow parameters
  selectedLotId = null,
  lotWorkflowActive = false,
  currentLot = null
) => {
// FIXED: Enhanced detection start with better lot workflow handling
const handleStartDetection = async () => {
  console.log("🚀 Detection start requested with context:", {
    cameraId,
    targetLabel,
    currentStreamingType,
    lotWorkflowActive,
    selectedLotId,
    hasCurrentLot: !!currentLot
  });

  // Basic validation
  if (!cameraId || cameraId === '') {
    showSnackbar("Please select a camera first.", "error");
    return { success: false, error: "No camera selected" };
  }
  
  if (!targetLabel || targetLabel.trim() === '') {
    showSnackbar("Please enter a target label for detection.", "error");
    return { success: false, error: "No target label" };
  }
  
  // FIXED: More intelligent lot validation for basic mode
  if (currentStreamingType === 'basic') {
    console.log("📋 Basic mode detection - checking lot requirements...");
    
    // If we're in lot workflow mode, we need EITHER currentLot OR selectedLotId
    if (lotWorkflowActive) {
      if (!selectedLotId && !currentLot) {
        console.log("❌ Lot workflow active but no lot ID or current lot available");
        showSnackbar("Lot workflow is active but no lot is selected. Please check the URL parameters.", "warning");
        return { success: false, error: "No lot in workflow" };
      }
      
      // FIXED: Allow detection to proceed if we have selectedLotId even without currentLot
      // The lot will be loaded automatically during detection
      console.log("✅ Lot workflow requirements satisfied - proceeding with detection");
    } else {
      // Not in lot workflow mode, we still need a currentLot for regular basic mode
      if (!currentLot) {
        console.log("❌ Basic mode requires a current lot when not in workflow mode");
        showSnackbar("Please create or select a lot before starting detection in basic mode.", "warning");
        return { success: false, error: "No current lot" };
      }
      
      console.log("✅ Basic mode lot requirements satisfied (current lot available)");
    }
  }
  
  // For optimized mode, no lot validation needed
  if (currentStreamingType === 'optimized') {
    console.log("✅ Optimized mode - no lot requirements");
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
    if (!proceed) {
      return { success: false, error: "Health check failed" };
    }
  }
  
  // Validate camera still exists
  const cameraExists = cameras.some(cam => cam.id.toString() === cameraId.toString());
  if (!cameraExists) {
    showSnackbar("Selected camera is no longer available. Please detect cameras and select a new one.", "error");
    return { success: false, error: "Camera not available" };
  }
  
  console.log(`✅ Starting ${currentStreamingType} detection with options:`, detectionOptions);
  
  // Additional logging for lot workflow
  if (lotWorkflowActive && selectedLotId) {
    console.log(`🎯 Detection starting in lot workflow mode for lot ${selectedLotId}`);
  } else if (currentLot) {
    console.log(`📦 Detection starting with current lot: ${currentLot.lot_name} (ID: ${currentLot.lot_id})`);
  }

  return { success: true };
};
  // Enhanced detection stop
  const handleStopDetection = async () => {
    console.log(`🛑 Stopping ${currentStreamingType} detection...`);
    
    // Reset basic mode states
    setIsStreamFrozen(false);
    setLastDetectionResult(null);
    setOnDemandDetecting(false);
    setDetectionInProgress(false);
    
    // Only clear current lot if we're not in lot workflow mode
    if (!lotWorkflowActive) {
      console.log("🗑️ Clearing current lot (not in workflow mode)");
      setCurrentLot(null);
    } else {
      console.log("📋 Keeping current lot (in workflow mode)");
    }
    
    // Perform post-shutdown health check after shutdown to verify clean state
    setTimeout(async () => {
      console.log("🩺 Checking system health after detection stop...");
      await performPostShutdownHealthCheck();
    }, 3000); // Wait 3 seconds for shutdown to complete

    return { success: true };
  };

  return {
    handleStartDetection,
    handleStopDetection
  };
};

// Input validation and change handlers
export const createInputHandlers = (
  cameras,
  showSnackbar,
  setSelectedCameraId,
  setCameraId,
  setCurrentLot,
  setLastDetectionResult,
  setIsStreamFrozen,
  setTargetLabel,
  setDetectionOptions,
  detectionState,
  // NEW: Lot workflow parameters
  lotWorkflowActive = false
) => {
  // Handle camera selection with validation
  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    console.log("Camera selection changed:", selectedCameraId);
    
    const cameraExists = cameras.some(cam => cam.id.toString() === selectedCameraId.toString());
    if (cameraExists || selectedCameraId === '') {
      setSelectedCameraId(selectedCameraId);
      setCameraId(selectedCameraId);
      
      // FIXED: Only reset lot-related state if we're not in lot workflow mode
      if (!lotWorkflowActive) {
        setCurrentLot(null);
        setLastDetectionResult(null);
        setIsStreamFrozen(false);
      }
    } else {
      console.warn("Selected camera not found in available cameras");
      showSnackbar("Selected camera is not available. Please choose a different camera.", "warning");
    }
  };

  // Handle target label changes with validation
  const handleTargetLabelChange = (event) => {
    const value = event.target.value;
    const sanitizedValue = value.replace(/[<>"/\\&]/g, '');
    setTargetLabel(sanitizedValue);
  };

  // Handle detection options changes
  const handleDetectionOptionsChange = (newOptions) => {
    setDetectionOptions(prev => ({
      ...prev,
      ...newOptions
    }));
    console.log("Detection options updated:", newOptions);
  };

  return {
    handleCameraChange,
    handleTargetLabelChange,
    handleDetectionOptionsChange
  };
};

// System state and info helpers
export const getStateInfo = (detectionState, currentStreamingType) => {
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
export const getModeDisplayInfo = (currentStreamingType) => {
  const isBasic = currentStreamingType === 'basic';
  return {
    icon: isBasic ? 'Smartphone' : 'Computer',
    color: isBasic ? 'warning' : 'success',
    description: isBasic 
      ? 'Lot-Based Detection - Tracked detection with database integration' 
      : 'Real-Time Detection - Continuous video stream analysis'
  };
};

// Performance status color helper
export const getPerformanceColor = (value, thresholds) => {
  if (value < thresholds.good) return 'success';
  if (value < thresholds.warning) return 'warning';
  return 'error';
};

// Cleanup function
export const createCleanupFunction = (
  cleanupRef,
  stopMonitoring,
  detectionState,
  cameraService
) => {
  return async (event) => {
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
};