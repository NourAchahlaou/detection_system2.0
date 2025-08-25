// utils/AppIdentificationUtils.js - Updated identification utilities with group support
import { identificationService } from "../service/MainIdentificationService";

// Identification states from service
export const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// System initialization functions
// FIXED: System initialization with proper health check flow and safer error handling
export const initializeIdentificationSystem = async (
  initializationAttempted,
  setInitializationError,
  performInitialHealthCheck,
  loadAvailableGroups
) => {
  if (initializationAttempted.current) {
    console.log("🔄 Initialization already attempted, skipping");
    return;
  }
  
  initializationAttempted.current = true;

  try {
    console.log("🚀 Starting identification system initialization...");
    
    // Step 1: Basic processor initialization (without group)
    const response = await fetch('/api/detection/identification/initialize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const initResult = await response.json();
    
    if (initResult.success) {
      console.log("✅ Basic identification processor initialized:", initResult.message);
      setInitializationError(null);
      
      // Step 2: Perform initial health check (non-blocking)
      try {
        console.log("🩺 Performing initial health check...");
        await performInitialHealthCheck();
        console.log("✅ Initial health check completed successfully");
      } catch (healthError) {
        console.warn("⚠️ Health check had issues but continuing:", healthError.message);
        // Health check failure is not fatal for initialization
      }
      
      // Step 3: Load available groups (non-blocking)
      try {
        console.log("📋 Loading available groups...");
        await loadAvailableGroups();
        console.log("✅ Available groups loaded successfully");
      } catch (groupError) {
        console.warn("⚠️ Group loading had issues but continuing:", groupError.message);
        // Group loading failure is not fatal for basic initialization
      }
      
      console.log("✅ System initialization completed successfully");
      return { success: true };
    } else {
      throw new Error(initResult.message || initResult.detail || 'Failed to initialize identification processor');
    }
    
  } catch (error) {
    console.error("❌ Error initializing identification system:", error);
    const errorMessage = error.message || 'Unknown initialization error';
    setInitializationError(errorMessage);
    
    // Reset the flag so retry can work
    initializationAttempted.current = false;
    
    throw new Error(errorMessage);
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

// ===================
// GROUP MANAGEMENT HANDLERS (NEW)
// ===================

export const createGroupManagementHandlers = (
  targetGroupName,
  currentGroup,
  availableGroups,
  showSnackbar,
  setTargetGroupName,
  setAvailableGroups,
  setCurrentGroup,
  setIsGroupLoaded,
  setAvailablePieceTypes
) => {
  // Load available groups - FIXED API ENDPOINT
  const handleLoadAvailableGroups = async () => {
    try {
      console.log('📋 Loading available groups...');
      
      // Use the correct API endpoint from the router
      const response = await fetch('/api/detection/identification/groups/available');
      const result = await response.json();
      
      if (result.success) {
        setAvailableGroups(result.available_groups || []);
        setCurrentGroup(result.current_group);
        console.log(`✅ Loaded ${result.available_groups?.length || 0} available groups, current: '${result.current_group}'`);
        return { success: true, groups: result.available_groups };
      } else {
        console.warn('📋 Failed to load groups:', result.message);
        setAvailableGroups([]);
        return { success: false, error: result.message };
      }
    } catch (error) {
      console.error('❌ Error loading groups:', error);
      showSnackbar(`Failed to load groups: ${error.message}`, 'error');
      setAvailableGroups([]);
      return { success: false, error: error.message };
    }
  };


  // Select a group
  const handleSelectGroup = async (groupName) => {
    if (!groupName?.trim()) {
      showSnackbar('Please enter a group name', 'warning');
      return { success: false, error: 'No group name provided' };
    }

    try {
      console.log(`🔄 Selecting group: ${groupName}`);
      
      // Use the correct API endpoint from the router
      const response = await fetch(`/api/detection/identification/groups/select/${encodeURIComponent(groupName.trim())}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();
      
      if (result.success) {
        setCurrentGroup(result.current_group);
        setIsGroupLoaded(result.is_group_loaded);
        setTargetGroupName(groupName.trim());
        
        // Load piece types for this group
        await handleGetPieceTypesForGroup(groupName.trim());
        
        showSnackbar(`Group "${result.current_group}" loaded successfully`, 'success');
        return { success: true, group: result.current_group };
      } else {
        throw new Error(result.detail || result.message || 'Failed to select group');
      }
    } catch (error) {
      console.error('❌ Error selecting group:', error);
      showSnackbar(`Failed to select group "${groupName}": ${error.message}`, 'error');
      setIsGroupLoaded(false);
      setCurrentGroup(null);
      return { success: false, error: error.message };
    }
  };

  // Handle target group name change
  const handleTargetGroupNameChange = (event) => {
    const groupName = event.target.value;
    setTargetGroupName(groupName);
    console.log(`📝 Group name changed to: ${groupName}`);
  };

  const handleGetPieceTypesForGroup = async (groupName) => {
    if (!groupName?.trim()) {
      return { success: false, error: 'No group name provided' };
    }

    try {
      console.log(`📋 Loading piece types for group: ${groupName}`);
      
      // Use the correct API endpoint from the router
      const response = await fetch(`/api/detection/identification/groups/${encodeURIComponent(groupName)}/piece-types`);
      const result = await response.json();
      
      if (result.success) {
        setAvailablePieceTypes(result.available_piece_types || []);
        console.log(`✅ Loaded ${result.available_piece_types?.length || 0} piece types for group ${groupName}`);
        return { 
          success: true, 
          pieceTypes: result.available_piece_types,
          totalClasses: result.total_classes
        };
      } else {
        console.warn('📋 Failed to load piece types:', result.message);
        setAvailablePieceTypes([]);
        return { success: false, error: result.message };
      }
    } catch (error) {
      console.error('❌ Error loading piece types:', error);
      setAvailablePieceTypes([]);
      return { success: false, error: error.message };
    }
  };

  // Switch group and perform identification - FIXED API ENDPOINT
  const handleSwitchGroupAndIdentify = async (cameraId, newGroupName, options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for identification", "error");
      return { success: false, error: "No camera ID provided" };
    }

    if (!newGroupName?.trim()) {
      showSnackbar("Group name is required for identification", "error");
      return { success: false, error: "No group name provided" };
    }

    try {
      console.log(`🔄 Switching to group '${newGroupName}' and identifying for camera ${cameraId}`);
      
      // Use the correct API endpoint from the router
      const response = await fetch(`/api/detection/identification/identify/${cameraId}/switch-group`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          new_group_name: newGroupName.trim(),
          freeze_stream: options.freezeStream !== false,
          quality: options.quality || 85
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setCurrentGroup(result.current_group);
        setTargetGroupName(newGroupName.trim());
        setIsGroupLoaded(true);
        
        showSnackbar(`Switched to group "${result.current_group}" and identified ${result.summary.total_pieces} pieces`, 'success');
        return { success: true, result: result };
      } else {
        throw new Error(result.detail || result.message || 'Group switch and identification failed');
      }
    } catch (error) {
      console.error('❌ Error in group switch and identification:', error);
      showSnackbar(`Group switch and identification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  return {
    handleLoadAvailableGroups,
    handleSelectGroup,
    handleTargetGroupNameChange,
    handleGetPieceTypesForGroup,
    handleSwitchGroupAndIdentify
  };
};

// ===================
// IDENTIFICATION CONTROL HANDLERS (Updated for Groups)
// ===================

export const createIdentificationControlHandlers = (
  cameraId,
  identificationState,
  systemHealth,
  cameras,
  targetGroupName,
  showSnackbar,
  performSingleHealthCheck,
  lastHealthCheck,
  setIsStreamFrozen,
  setLastIdentificationResult,
  setIdentificationInProgress,
  performPostShutdownHealthCheck
) => {
  // Enhanced identification start with group validation
  const handleStartIdentification = async () => {
    console.log("🚀 Identification start requested with context:", {
      cameraId,
      identificationState,
      targetGroupName
    });

    // Basic validation
    if (!cameraId || cameraId === '') {
      showSnackbar("Please select a camera first.", "error");
      return { success: false, error: "No camera selected" };
    }

    // Group validation
    if (!targetGroupName || targetGroupName.trim() === '') {
      showSnackbar("Please enter a target group name first.", "error");
      return { success: false, error: "No group selected" };
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
    
    console.log(`✅ Starting identification for camera ${cameraId} with group ${targetGroupName}`);
    return { success: true, cameraId, groupName: targetGroupName.trim() };
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

// ===================
// IDENTIFICATION STREAM HANDLERS (Updated for Groups)
// ===================

export const createIdentificationStreamHandlers = (
  cameraId,
  identificationInProgress,
  confidenceThreshold,
  targetGroupName,
  showSnackbar,
  setIdentificationInProgress,
  setLastIdentificationResult,
  setIsStreamFrozen
) => {
  // Handle piece identification with group support
  const handlePieceIdentification = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for identification", "error");
      return { success: false, error: "No camera ID provided" };
    }

    // Use provided group name or fall back to targetGroupName
    const groupName = options.groupName || targetGroupName;
    if (!groupName || groupName.trim() === '') {
      showSnackbar("Group name is required for identification", "error");
      return { success: false, error: "No group name provided" };
    }

    if (identificationInProgress) {
      console.log("🚫 Identification already in progress, skipping...");
      return { success: false, error: "Identification in progress" };
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing piece identification for camera ${cameraId}, group ${groupName}`);
      
      const response = await fetch(`/api/detection/identification/identify/${cameraId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          group_name: groupName.trim(),
          freeze_stream: options.freezeStream !== false,
          quality: options.quality || 85
        })
      });

      const result = await response.json();

      if (result.success) {
        setLastIdentificationResult({
          summary: result.summary,
          pieces: result.identification_result?.pieces || result.pieces,
          processingTime: result.processing_time_ms,
          frameWithOverlay: result.frame_with_overlay,
          streamFrozen: result.stream_frozen,
          timestamp: result.timestamp,
          message: result.message,
          groupName: result.group_name
        });
        
        // Update stream frozen state
        setIsStreamFrozen(result.stream_frozen);

        const message = result.summary.total_pieces > 0 
          ? `🎯 Identified ${result.summary.total_pieces} pieces from group "${groupName}" (${result.summary.unique_labels} unique types)!`
          : `No pieces identified for group "${groupName}". Try adjusting the camera angle or lighting.`;

        showSnackbar(message, result.summary.total_pieces > 0 ? 'success' : 'info');
        
        return {
          success: true,
          identificationResult: result
        };
      } else {
        throw new Error(result.detail || result.message || 'Identification failed');
      }
    } catch (error) {
      console.error('❌ Error in piece identification:', error);
      showSnackbar(`Identification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Handle quick analysis with group support
  const handleQuickAnalysis = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for analysis", "error");
      return { success: false, error: "No camera ID provided" };
    }

    // Use provided group name or fall back to targetGroupName
    const groupName = options.groupName || targetGroupName;
    if (!groupName || groupName.trim() === '') {
      showSnackbar("Group name is required for analysis", "error");
      return { success: false, error: "No group name provided" };
    }

    if (identificationInProgress) {
      console.log("🚫 Analysis already in progress, skipping...");
      return { success: false, error: "Analysis in progress" };
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing quick analysis for camera ${cameraId}, group ${groupName}`);
      
      const response = await fetch(`/api/identification/analyze/${cameraId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          group_name: groupName.trim(),
          quality: options.quality || 85
        })
      });

      const result = await response.json();

      if (result.success) {
        setLastIdentificationResult({
          piecesFound: result.pieces_found,
          pieces: result.pieces,
          summary: result.summary,
          processingTime: result.processing_time_ms,
          timestamp: result.timestamp,
          message: result.message,
          isQuickAnalysis: true,
          groupName: result.group_name
        });

        const message = result.pieces_found > 0 
          ? `🔍 Quick analysis found ${result.pieces_found} pieces from group "${groupName}"!`
          : `No pieces found in quick analysis for group "${groupName}".`;

        showSnackbar(message, result.pieces_found > 0 ? 'success' : 'info');
        
        return {
          success: true,
          analysisResult: result
        };
      } else {
        throw new Error(result.detail || result.message || 'Quick analysis failed');
      }
    } catch (error) {
      console.error('❌ Error in quick analysis:', error);
      showSnackbar(`Quick analysis failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Handle batch identification with group support
  const handleBatchIdentification = async (options = {}) => {
    if (!cameraId) {
      showSnackbar("Camera ID is required for batch identification", "error");
      return { success: false, error: "No camera ID provided" };
    }

    // Use provided group name or fall back to targetGroupName
    const groupName = options.groupName || targetGroupName;
    if (!groupName || groupName.trim() === '') {
      showSnackbar("Group name is required for batch identification", "error");
      return { success: false, error: "No group name provided" };
    }

    if (identificationInProgress) {
      console.log("🚫 Batch identification already in progress, skipping...");
      return { success: false, error: "Batch identification in progress" };
    }

    setIdentificationInProgress(true);

    try {
      console.log(`📦 Performing batch identification for camera ${cameraId}, group ${groupName}`);
      
      const response = await fetch(`/api/identification/batch/${cameraId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          group_name: groupName.trim(),
          num_frames: options.numFrames || 5,
          interval_seconds: options.intervalSeconds || 1.0
        })
      });

      const result = await response.json();

      if (result.success) {
        setLastIdentificationResult({
          batchResult: result.batch_identification_result,
          framesProcessed: result.frames_processed,
          totalPiecesFound: result.total_pieces_found,
          uniqueLabels: result.unique_labels,
          averagePiecesPerFrame: result.average_pieces_per_frame,
          averageConfidence: result.average_confidence,
          timestamp: result.timestamp,
          message: result.message,
          isBatchIdentification: true,
          groupName: result.group_name
        });

        const message = result.total_pieces_found > 0 
          ? `📦 Batch identification completed for group "${groupName}": ${result.total_pieces_found} total pieces across ${result.frames_processed} frames!`
          : `No pieces found in batch identification for group "${groupName}".`;

        showSnackbar(message, result.total_pieces_found > 0 ? 'success' : 'info');
        
        return {
          success: true,
          batchResult: result
        };
      } else {
        throw new Error(result.detail || result.message || 'Batch identification failed');
      }
    } catch (error) {
      console.error('❌ Error in batch identification:', error);
      showSnackbar(`Batch identification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Freeze stream (unchanged)
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

  // Unfreeze stream (unchanged)
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
    handleBatchIdentification,
    handleFreezeStream,
    handleUnfreezeStream
  };
};

// Input validation and change handlers (unchanged)
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

// ===================
// VALIDATION UTILITIES (Updated for Groups)
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

  // Validate group name
  const validateGroupName = (groupName, availableGroups = []) => {
    if (!groupName || groupName.trim() === '') {
      return { valid: false, error: 'Group name is required' };
    }

    if (availableGroups.length > 0 && !availableGroups.includes(groupName.trim())) {
      return { 
        valid: false, 
        error: `Group "${groupName}" not found. Available: ${availableGroups.join(', ')}` 
      };
    }

    return { valid: true, value: groupName.trim() };
  };

  // Validate identification options with group
  const validateIdentificationOptions = (options = {}) => {
    const validOptions = {};
    
    if (options.groupName !== undefined) {
      const groupValidation = validateGroupName(options.groupName);
      if (!groupValidation.valid) {
        return { valid: false, error: groupValidation.error };
      }
      validOptions.groupName = groupValidation.value;
    }

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

    if (options.numFrames !== undefined) {
      const numFrames = parseInt(options.numFrames);
      if (isNaN(numFrames) || numFrames < 1 || numFrames > 20) {
        return { valid: false, error: 'Number of frames must be between 1 and 20' };
      }
      validOptions.numFrames = numFrames;
    }

    if (options.intervalSeconds !== undefined) {
      const interval = parseFloat(options.intervalSeconds);
      if (isNaN(interval) || interval < 0.1 || interval > 10.0) {
        return { valid: false, error: 'Interval must be between 0.1 and 10.0 seconds' };
      }
      validOptions.intervalSeconds = interval;
    }

    return { valid: true, options: validOptions };
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

  return {
    validateCameraId,
    validateGroupName,
    validateConfidenceThreshold,
    validateIdentificationOptions
  };
};

// Stats handlers (unchanged but with group context)
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

// Stream management handlers (unchanged)
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

// Status and info utilities (updated with group context)
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

// Settings and configuration utilities (updated for groups)
export const createSettingsHandlers = (
  showSnackbar,
  setConfidenceThreshold,
  setAvailablePieceTypes
) => {
  // Update confidence threshold
  const handleUpdateConfidenceThreshold = async (threshold) => {
    try {
      console.log(`🎯 Updating confidence threshold to ${threshold}...`);
      
      const response = await fetch('/api/identification/settings/confidence-threshold', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ threshold: parseFloat(threshold) })
      });

      const result = await response.json();
      
      if (result.success) {
        setConfidenceThreshold(result.new_threshold);
        showSnackbar(`Confidence threshold updated to ${result.new_threshold}`, 'success');
        return { success: true, threshold: result.new_threshold };
      } else {
        throw new Error(result.detail || result.message || 'Failed to update threshold');
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
      const response = await fetch('/api/identification/settings');
      const result = await response.json();
      
      if (result.success) {
        return { success: true, settings: result.current_settings };
      } else {
        throw new Error(result.detail || result.message || 'Failed to get settings');
      }
    } catch (error) {
      console.error('Error getting identification settings:', error);
      showSnackbar(`Failed to get settings: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  };

  // Load available piece types for current group
  const handleLoadAvailablePieceTypes = async (groupName = null) => {
    try {
      let response;
      if (groupName) {
        response = await fetch(`/api/identification/groups/${encodeURIComponent(groupName)}/piece-types`);
      } else {
        // Get general piece types - this might not be applicable with group-based system
        response = await fetch('/api/identification/settings');
        const settingsResult = await response.json();
        if (settingsResult.success && settingsResult.model_info) {
          setAvailablePieceTypes(Object.values(settingsResult.model_info.class_names || {}));
          return { 
            success: true, 
            pieceTypes: Object.values(settingsResult.model_info.class_names || {}) 
          };
        }
        return { success: false, error: 'No group specified and no general piece types available' };
      }
      
      const result = await response.json();
      
      if (result.success) {
        setAvailablePieceTypes(result.available_piece_types || []);
        return { 
          success: true, 
          pieceTypes: result.available_piece_types,
          totalClasses: result.total_classes
        };
      } else {
        throw new Error(result.detail || result.message || 'Failed to load piece types');
      }
    } catch (error) {
      console.error('Error loading piece types:', error);
      showSnackbar(`Failed to load piece types: ${error.message}`, 'error');
      setAvailablePieceTypes([]);
      return { success: false, error: error.message };
    }
  };

  return {
    handleUpdateConfidenceThreshold,
    handleGetIdentificationSettings,
    handleLoadAvailablePieceTypes
  };
};

// Cleanup and maintenance utilities (unchanged)
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

// Cleanup function (updated)
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
      
      // Stop all identification services
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

// System state and info helpers with group context
export const getStateInfo = (identificationState, currentGroup = null) => {
  const groupInfo = currentGroup ? ` for group "${currentGroup}"` : '';
  
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
        message: `System ready for identification${groupInfo}`,
        canOperate: true
      };
    case IdentificationStates.RUNNING:
      return {
        color: 'warning',
        message: `IDENTIFICATION active${groupInfo}`,
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

// Helper function to validate group-based identification readiness
export const validateIdentificationReadiness = (
  identificationState,
  cameraId,
  targetGroupName,
  availableGroups = []
) => {
  const issues = [];
  
  // Check system state
  if (identificationState !== IdentificationStates.READY) {
    issues.push(`System not ready (current state: ${identificationState})`);
  }
  
  // Check camera selection
  if (!cameraId || cameraId === '') {
    issues.push('No camera selected');
  }
  
  // Check group selection
  if (!targetGroupName || targetGroupName.trim() === '') {
    issues.push('No group selected');
  } else if (availableGroups.length > 0 && !availableGroups.includes(targetGroupName.trim())) {
    issues.push(`Selected group "${targetGroupName}" is not available`);
  }
  
  return {
    isReady: issues.length === 0,
    issues: issues,
    canProceed: issues.length === 0
  };
};

// Helper function to get group status info
export const getGroupStatusInfo = (
  currentGroup,
  targetGroupName,
  availableGroups = [],
  isGroupLoaded = false
) => {
  if (!currentGroup && !targetGroupName) {
    return {
      status: 'no_group',
      message: 'No group selected',
      color: 'warning',
      canIdentify: false
    };
  }
  
  if (targetGroupName && !isGroupLoaded) {
    return {
      status: 'loading',
      message: `Loading group "${targetGroupName}"...`,
      color: 'info',
      canIdentify: false
    };
  }
  
  if (currentGroup && isGroupLoaded) {
    return {
      status: 'ready',
      message: `Group "${currentGroup}" loaded and ready`,
      color: 'success',
      canIdentify: true
    };
  }
  
  if (targetGroupName && availableGroups.length > 0 && !availableGroups.includes(targetGroupName)) {
    return {
      status: 'invalid',
      message: `Group "${targetGroupName}" not found`,
      color: 'error',
      canIdentify: false
    };
  }
  
  return {
    status: 'unknown',
    message: 'Group status unknown',
    color: 'default',
    canIdentify: false
  };
};