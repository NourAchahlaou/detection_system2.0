// utils/AppIdentificationUtils.js - Simplified identification utilities
import { identificationService } from "../service/MainIdentificationService";

// Identification states from service
export const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// System initialization functions
export const initializeIdentificationSystem = async (
  initializationAttempted,
  setInitializationError,
  performInitialHealthCheck,
  loadAvailablePieceTypes
) => {
  if (initializationAttempted.current) return;
  initializationAttempted.current = true;

  try {
    console.log("🚀 Starting identification system initialization...");
    
    // Initialize identification processor
    const initResult = await identificationService.ensureInitialized();
    
    if (initResult.success) {
      console.log("✅ Identification system initialized:", initResult.message);
      setInitializationError(null);
      
      // Perform initial health check right after initialization
      await performInitialHealthCheck();
      
      // Load available piece types
      await loadAvailablePieceTypes();
      
      console.log("✅ System initialization completed successfully");
    } else {
      throw new Error(initResult.message || 'Failed to initialize identification system');
    }
    
  } catch (error) {
    console.error("❌ Error initializing identification system:", error);
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
      identificationService.resetToInitializing('Manual retry');
      
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

// Identification control functions
export const createIdentificationControlHandlers = (
  cameraId,
  identificationState,
  systemHealth,
  cameras,
  showSnackbar,
  performSingleHealthCheck,
  lastHealthCheck,
  setIsStreamFrozen,
  setLastIdentificationResult,
  setIdentificationInProgress,
  performPostShutdownHealthCheck
) => {
  // Enhanced identification start with validation
  const handleStartIdentification = async () => {
    console.log("🚀 Identification start requested with context:", {
      cameraId,
      identificationState
    });

    // Basic validation
    if (!cameraId || cameraId === '') {
      showSnackbar("Please select a camera first.", "error");
      return { success: false, error: "No camera selected" };
    }
    
    // Perform health check before starting identification if not done recently
    const timeSinceLastCheck = lastHealthCheck.current ? Date.now() - lastHealthCheck.current : Infinity;
    if (timeSinceLastCheck > 30000 || !systemHealth.overall) { // 30 seconds
      console.log("🩺 Checking system health before starting identification...");
      await performSingleHealthCheck();
    }
    
    if (!systemHealth.overall) {
      const proceed = window.confirm(
        `System health check indicates issues. Do you want to proceed anyway?`
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
    
    console.log(`✅ Starting identification for camera ${cameraId}`);
    return { success: true };
  };

  // Enhanced identification stop
  const handleStopIdentification = async () => {
    console.log(`🛑 Stopping identification...`);
    
    // Reset identification states
    setIsStreamFrozen(false);
    setLastIdentificationResult(null);
    setIdentificationInProgress(false);
    
    // Perform post-shutdown health check after shutdown to verify clean state
    setTimeout(async () => {
      console.log("🩺 Checking system health after identification stop...");
      await performPostShutdownHealthCheck();
    }, 3000); // Wait 3 seconds for shutdown to complete

    return { success: true };
  };

  return {
    handleStartIdentification,
    handleStopIdentification
  };
};

// Identification stream handlers
export const createIdentificationStreamHandlers = (
  cameraId,
  identificationInProgress,
  confidenceThreshold,
  showSnackbar,
  setIdentificationInProgress,
  setLastIdentificationResult,
  setIsStreamFrozen
) => {
  // Handle piece identification
  const handlePieceIdentification = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for identification", "error");
      return { success: false };
    }

    if (identificationInProgress) {
      console.log("🚫 Identification already in progress, skipping...");
      return { success: false, error: "Identification in progress" };
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing piece identification for camera ${cameraId}`);
      
      const result = await identificationService.performPieceIdentification(cameraId, {
        freezeStream: options.freezeStream !== false, // Default to true
        quality: options.quality || 85,
        ...options
      });

      if (result.success) {
        setLastIdentificationResult({
          summary: result.summary,
          pieces: result.pieces,
          processingTime: result.processingTime,
          frameWithOverlay: result.frameWithOverlay,
          streamFrozen: result.streamFrozen,
          timestamp: result.timestamp,
          message: result.message
        });
        
        // Update stream frozen state
        setIsStreamFrozen(result.streamFrozen);

        const message = result.summary.total_pieces > 0 
          ? `🎯 Identified ${result.summary.total_pieces} pieces (${result.summary.unique_labels} unique types)!`
          : `No pieces identified. Try adjusting the camera angle or lighting.`;

        showSnackbar(message, result.summary.total_pieces > 0 ? 'success' : 'info');
        
        return {
          success: true,
          identificationResult: result
        };
      } else {
        throw new Error(result.message || 'Identification failed');
      }
    } catch (error) {
      console.error('❌ Error in piece identification:', error);
      showSnackbar(`Identification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Handle quick analysis
  const handleQuickAnalysis = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for analysis", "error");
      return { success: false };
    }

    if (identificationInProgress) {
      console.log("🚫 Analysis already in progress, skipping...");
      return { success: false, error: "Analysis in progress" };
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing quick analysis for camera ${cameraId}`);
      
      const result = await identificationService.performQuickAnalysis(cameraId, {
        analyzeFrameOnly: options.analyzeFrameOnly !== false, // Default to true
        quality: options.quality || 85,
        ...options
      });

      if (result.success) {
        setLastIdentificationResult({
          piecesFound: result.piecesFound,
          pieces: result.pieces,
          summary: result.summary,
          processingTime: result.processingTime,
          timestamp: result.timestamp,
          message: result.message,
          isQuickAnalysis: true
        });

        const message = result.piecesFound > 0 
          ? `🔍 Quick analysis found ${result.piecesFound} pieces!`
          : `No pieces found in quick analysis.`;

        showSnackbar(message, result.piecesFound > 0 ? 'success' : 'info');
        
        return {
          success: true,
          analysisResult: result
        };
      } else {
        throw new Error(result.message || 'Quick analysis failed');
      }
    } catch (error) {
      console.error('❌ Error in quick analysis:', error);
      showSnackbar(`Quick analysis failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Freeze stream
  const handleFreezeStream = async () => {
    if (!cameraId) return;

    try {
      console.log(`🧊 Freezing stream for camera ${cameraId}`);
      await identificationService.freezeStream(cameraId);
      console.log(`✅ Stream frozen for camera ${cameraId}`);
    } catch (error) {
      console.error("❌ Error freezing stream:", error);
      showSnackbar(`Failed to freeze stream: ${error.message}`, "error");
    }
  };

  // Unfreeze stream
  const handleUnfreezeStream = async () => {
    if (!cameraId) return;

    try {
      console.log(`🔥 Unfreezing stream for camera ${cameraId}`);
      await identificationService.unfreezeStream(cameraId);
      console.log(`✅ Stream unfrozen for camera ${cameraId}`);
    } catch (error) {
      console.error("❌ Error unfreezing stream:", error);
      showSnackbar(`Failed to unfreeze stream: ${error.message}`, "error");
    }
  };

  return {
    handlePieceIdentification,
    handleQuickAnalysis,
    handleFreezeStream,
    handleUnfreezeStream
  };
};

// Input validation and change handlers
export const createInputHandlers = (
  cameras,
  showSnackbar,
  setSelectedCameraId,
  setCameraId,
  setLastIdentificationResult,
  setIsStreamFrozen,
  identificationState
) => {
  // Handle camera selection with validation
  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    console.log("Camera selection changed:", selectedCameraId);
    
    const cameraExists = cameras.some(cam => cam.id.toString() === selectedCameraId.toString());
    if (cameraExists || selectedCameraId === '') {
      setSelectedCameraId(selectedCameraId);
      setCameraId(selectedCameraId);
      
      // Reset identification-related state when camera changes
      setLastIdentificationResult(null);
      setIsStreamFrozen(false);
    } else {
      console.warn("Selected camera not found in available cameras");
      showSnackbar("Selected camera is not available. Please choose a different camera.", "warning");
    }
  };

  return {
    handleCameraChange
  };
};

export const createStatsHandlers = (
  cameraId,
  showSnackbar
) => {
  // Get identification stats for specific camera
  const handleGetCameraStats = async () => {
    try {
      const stats = identificationService.getIdentificationStatsForCamera(cameraId);
      return { success: true, stats };
    } catch (error) {
      console.error('Error getting camera stats:', error);
      showSnackbar(`Failed to get stats: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Get all streaming stats
  const handleGetAllStats = async () => {
    try {
      const stats = await identificationService.getAllStreamingStats();
      return { success: true, stats };
    } catch (error) {
      console.error('Error getting all stats:', error);
      showSnackbar(`Failed to get all stats: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Get identification history
  const handleGetIdentificationHistory = async () => {
    try {
      const result = await identificationService.getIdentificationHistory();
      if (result.success) {
        return { success: true, history: result.history };
      }
      throw new Error(result.message || 'Failed to get history');
    } catch (error) {
      console.error('Error getting identification history:', error);
      showSnackbar(`Failed to get history: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  return {
    handleGetCameraStats,
    handleGetAllStats,
    handleGetIdentificationHistory
  };
};

// ===================
// STREAM MANAGEMENT UTILITIES
// ===================

export const createStreamManagementHandlers = (
  cameraId,
  showSnackbar,
  setStreamInfo
) => {
  // Start identification stream
  const handleStartStream = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required to start stream", "error");
      return { success: false };
    }

    try {
      console.log(`🎥 Starting identification stream for camera ${cameraId}`);
      const result = await identificationService.startIdentificationStream(cameraId, options);
      
      if (result.success) {
        showSnackbar(`Stream started for camera ${cameraId}`, 'success');
        return { success: true, streamInfo: result };
      } else {
        throw new Error(result.message || 'Failed to start stream');
      }
    } catch (error) {
      console.error('Error starting stream:', error);
      showSnackbar(`Failed to start stream: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Stop identification stream
  const handleStopStream = async () => {
    if (!cameraId) {
      showSnackbar("Camera ID is required to stop stream", "error");
      return { success: false };
    }

    try {
      console.log(`🛑 Stopping identification stream for camera ${cameraId}`);
      const result = await identificationService.stopIdentificationStream(cameraId);
      
      if (result.success) {
        showSnackbar(`Stream stopped for camera ${cameraId}`, 'success');
        return { success: true };
      } else {
        throw new Error(result.message || 'Failed to stop stream');
      }
    } catch (error) {
      console.error('Error stopping stream:', error);
      showSnackbar(`Failed to stop stream: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Stop all streams
  const handleStopAllStreams = async (performShutdown = true) => {
    try {
      console.log('🛑 Stopping all identification streams...');
      const result = await identificationService.stopAllStreams(performShutdown);
      
      if (result.success) {
        showSnackbar('All streams stopped', 'success');
        return { success: true };
      } else {
        throw new Error(result.message || 'Failed to stop all streams');
      }
    } catch (error) {
      console.error('Error stopping all streams:', error);
      showSnackbar(`Failed to stop all streams: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Ensure camera is started
  const handleEnsureCameraStarted = async () => {
    if (!cameraId) {
      showSnackbar("Camera ID is required", "error");
      return { success: false };
    }

    try {
      console.log(`📹 Ensuring camera ${cameraId} is started...`);
      const result = await identificationService.ensureCameraStarted(cameraId);
      
      if (result.success) {
        return { success: true, cameraInfo: result };
      } else {
        throw new Error(result.message || 'Failed to ensure camera started');
      }
    } catch (error) {
      console.error('Error ensuring camera started:', error);
      showSnackbar(`Failed to ensure camera started: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  return {
    handleStartStream,
    handleStopStream,
    handleStopAllStreams,
    handleEnsureCameraStarted
  };
};

// ===================
// STATUS AND INFO UTILITIES
// ===================

export const createStatusHandlers = (
  showSnackbar
) => {
  // Get detailed system status
  const handleGetDetailedStatus = () => {
    try {
      const status = identificationService.getDetailedStatus();
      return { success: true, status };
    } catch (error) {
      console.error('Error getting detailed status:', error);
      showSnackbar(`Failed to get status: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Get stream information
  const handleGetStreamInfo = () => {
    try {
      const streamInfo = identificationService.getStreamInfo();
      return { success: true, streamInfo };
    } catch (error) {
      console.error('Error getting stream info:', error);
      showSnackbar(`Failed to get stream info: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Get camera stream info
  const handleGetCameraStreamInfo = (cameraId) => {
    try {
      const streamInfo = identificationService.getCameraStreamInfo(cameraId);
      return { success: true, streamInfo };
    } catch (error) {
      console.error('Error getting camera stream info:', error);
      showSnackbar(`Failed to get camera stream info: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Check if camera is active
  const handleIsCameraActive = (cameraId) => {
    try {
      const isActive = identificationService.isCameraActive(cameraId);
      return { success: true, isActive };
    } catch (error) {
      console.error('Error checking if camera is active:', error);
      return { success: false, error: error.message };
    }
  };

  return {
    handleGetDetailedStatus,
    handleGetStreamInfo,
    handleGetCameraStreamInfo,
    handleIsCameraActive
  };
};

// ===================
// SETTINGS AND CONFIGURATION UTILITIES
// ===================

export const createSettingsHandlers = (
  showSnackbar,
  setConfidenceThreshold,
  setAvailablePieceTypes
) => {
  // Update confidence threshold
  const handleUpdateConfidenceThreshold = async (threshold) => {
    try {
      console.log(`🎯 Updating confidence threshold to ${threshold}...`);
      const result = await identificationService.updateConfidenceThreshold(threshold);
      
      if (result.success) {
        setConfidenceThreshold(result.newThreshold);
        showSnackbar(`Confidence threshold updated to ${result.newThreshold}`, 'success');
        return { success: true, threshold: result.newThreshold };
      } else {
        throw new Error(result.message || 'Failed to update threshold');
      }
    } catch (error) {
      console.error('Error updating confidence threshold:', error);
      showSnackbar(`Failed to update threshold: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Get identification settings
  const handleGetIdentificationSettings = async () => {
    try {
      const result = await identificationService.getIdentificationSettings();
      
      if (result.success) {
        return { success: true, settings: result.settings };
      } else {
        throw new Error(result.message || 'Failed to get settings');
      }
    } catch (error) {
      console.error('Error getting identification settings:', error);
      showSnackbar(`Failed to get settings: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Load available piece types
  const handleLoadAvailablePieceTypes = async () => {
    try {
      const result = await identificationService.getAvailablePieceTypes();
      
      if (result.success) {
        setAvailablePieceTypes(result.availablePieceTypes || []);
        return { success: true, pieceTypes: result.availablePieceTypes };
      } else {
        throw new Error(result.message || 'Failed to load piece types');
      }
    } catch (error) {
      console.error('Error loading piece types:', error);
      showSnackbar(`Failed to load piece types: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  return {
    handleUpdateConfidenceThreshold,
    handleGetIdentificationSettings,
    handleLoadAvailablePieceTypes
  };
};

// ===================
// CLEANUP AND MAINTENANCE UTILITIES
// ===================

export const createMaintenanceHandlers = (
  showSnackbar
) => {
  // Cleanup service
  const handleCleanup = async () => {
    try {
      console.log('🧹 Performing identification service cleanup...');
      await identificationService.cleanup();
      showSnackbar('Cleanup completed successfully', 'success');
      return { success: true };
    } catch (error) {
      console.error('Error during cleanup:', error);
      showSnackbar(`Cleanup failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Reset to initializing state
  const handleResetToInitializing = (reason = 'Manual reset') => {
    try {
      console.log(`🔄 Resetting to initializing state: ${reason}`);
      identificationService.resetToInitializing(reason);
      showSnackbar('Service reset to initializing state', 'info');
      return { success: true };
    } catch (error) {
      console.error('Error resetting to initializing:', error);
      showSnackbar(`Reset failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  return {
    handleCleanup,
    handleResetToInitializing
  };
};

// ===================
// VALIDATION UTILITIES
// ===================

export const createValidationUtils = () => {
  // Validate camera ID
  const validateCameraId = (cameraId, cameras) => {
    if (!cameraId || cameraId === '') {
      return { valid: false, error: 'Camera ID is required' };
    }

    const cameraExists = cameras.some(cam => cam.id.toString() === cameraId.toString());
    if (!cameraExists) {
      return { valid: false, error: 'Selected camera is not available' };
    }

    return { valid: true };
  };

  // Validate confidence threshold
  const validateConfidenceThreshold = (threshold) => {
    const numThreshold = parseFloat(threshold);
    
    if (isNaN(numThreshold)) {
      return { valid: false, error: 'Confidence threshold must be a number' };
    }
    
    if (numThreshold < 0 || numThreshold > 1) {
      return { valid: false, error: 'Confidence threshold must be between 0 and 1' };
    }

    return { valid: true, value: numThreshold };
  };

  // Validate identification options
  const validateIdentificationOptions = (options = {}) => {
    const validOptions = {};
    
    if (options.quality !== undefined) {
      const quality = parseInt(options.quality);
      if (isNaN(quality) || quality < 1 || quality > 100) {
        return { valid: false, error: 'Quality must be between 1 and 100' };
      }
      validOptions.quality = quality;
    }

    if (options.freezeStream !== undefined) {
      validOptions.freezeStream = Boolean(options.freezeStream);
    }

    if (options.analyzeFrameOnly !== undefined) {
      validOptions.analyzeFrameOnly = Boolean(options.analyzeFrameOnly);
    }

    return { valid: true, options: validOptions };
  };

  return {
    validateCameraId,
    validateConfidenceThreshold,
    validateIdentificationOptions
  };
  
};

// Cleanup function
export const createCleanupFunction = (
  cleanupRef,
  identificationState,
  cameraService
) => {
  return async (event) => {
    if (cleanupRef.current) return;
    cleanupRef.current = true;
    
    try {
      console.log("Performing cleanup...");
      
      // Stop all detection services
      if (identificationState === IdentificationStates.RUNNING) {
        await identificationService.stopAllStreams();
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

// System state and info helpers - FIXED VERSION
export const getStateInfo = (identificationState, currentStreamingType = 'identification') => {
  switch (identificationState) {
    case IdentificationStates.INITIALIZING:
      return {
        color: 'info',
        message: 'Initializing identification system...',
        canOperate: false
      };
    case IdentificationStates.READY:
      return {
        color: 'success',
        message: `System ready for ${currentStreamingType}`,
        canOperate: true
      };
    case IdentificationStates.RUNNING:
      return {
        color: 'warning',
        message: `${currentStreamingType.toUpperCase()} active`,
        canOperate: true
      };
    case IdentificationStates.SHUTTING_DOWN:
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