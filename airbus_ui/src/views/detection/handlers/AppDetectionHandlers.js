// handlers/AppDetectionHandlers.js - FIXED: Enhanced event handlers with lot workflow support
import { useCallback } from "react";
import api from "../../../utils/UseAxios";
import {
  createSystemControlFunctions,
  createBasicModeHandlers,
  createDetectionControlHandlers,
  createInputHandlers
} from "../utils/AppDetectionUtils";

// Custom hook for all detection-related handlers with lot workflow support
export const useDetectionHandlers = ({
  // State values
  cameraId,
  targetLabel,
  currentLot,
  detectionInProgress,
  detectionOptions,
  detectionState,
  currentStreamingType,
  systemHealth,
  cameras,
  lastHealthCheck,
  
  // Lot workflow specific state
  selectedLotId,
  lotWorkflowActive,
  detectionHistory,
  
  // State setters
  setIsProfileRefreshing,
  setDetectionInProgress,
  setOnDemandDetecting,
  setLastDetectionResult,
  setIsStreamFrozen,
  setCurrentLot,
  setSelectedCameraId,
  setCameraId,
  setTargetLabel,
  setDetectionOptions,
  setDetectionHistory,
  
  // Functions
  showSnackbar,
  fetchExistingLots,
  performSingleHealthCheck,
  performPostShutdownHealthCheck,
  loadSelectedLot,
  
  // Enhanced functions
  streamManager,
  
  // Refs
  lastHealthCheck: lastHealthCheckRef
}) => {

  // ===== ENHANCED HANDLERS WITH LOT WORKFLOW SUPPORT =====

  // Create system control functions
  const systemControlFunctions = createSystemControlFunctions(
    detectionState,
    showSnackbar,
    currentStreamingType
  );

  // FIXED: Create basic mode handlers with lot workflow parameters
  const basicModeHandlers = createBasicModeHandlers(
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
    // NEW: Lot workflow parameters
    selectedLotId,
    lotWorkflowActive
  );

  // FIXED: Create detection control handlers with lot workflow parameters
  const detectionControlHandlers = createDetectionControlHandlers(
    cameraId,
    targetLabel,
    currentStreamingType,
    systemHealth,
    cameras,
    detectionOptions,
    showSnackbar,
    performSingleHealthCheck,
    lastHealthCheckRef,
    setIsStreamFrozen,
    setLastDetectionResult,
    setOnDemandDetecting,
    setDetectionInProgress,
    setCurrentLot,
    performPostShutdownHealthCheck,
    // NEW: Lot workflow parameters
    selectedLotId,
    lotWorkflowActive,
    currentLot
  );

  // FIXED: Create input handlers with lot workflow parameters
  const inputHandlers = createInputHandlers(
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
    // NEW: Lot workflow parameter
    lotWorkflowActive
  );

  // Wrapped system control handlers with proper dependencies
  const handleRefreshSystemProfile = useCallback(async () => {
    await systemControlFunctions.handleRefreshSystemProfile(setIsProfileRefreshing);
  }, [systemControlFunctions, setIsProfileRefreshing]);

  const handleSwitchToBasicMode = useCallback(async () => {
    await systemControlFunctions.handleSwitchToBasicMode();
  }, [systemControlFunctions]);

  const handleSwitchToOptimizedMode = useCallback(async () => {
    await systemControlFunctions.handleSwitchToOptimizedMode();
  }, [systemControlFunctions]);

  const handleEnableAutoMode = useCallback(async () => {
    await systemControlFunctions.handleEnableAutoMode();
  }, [systemControlFunctions]);

  const handleRunPerformanceTest = useCallback(async () => {
    await systemControlFunctions.handleRunPerformanceTest();
  }, [systemControlFunctions]);

  // Wrapped basic mode handlers
  const handleOnDemandDetection = useCallback(async (options = {}) => {
    await basicModeHandlers.handleOnDemandDetection(options);
  }, [basicModeHandlers]);

  const handleFreezeStream = useCallback(async () => {
    await basicModeHandlers.handleFreezeStream();
  }, [basicModeHandlers]);

  const handleUnfreezeStream = useCallback(async () => {
    await basicModeHandlers.handleUnfreezeStream();
  }, [basicModeHandlers]);

  // FIXED: Wrapped detection control handlers with proper error handling
  const handleStartDetection = useCallback(async () => {
    const result = await detectionControlHandlers.handleStartDetection();
    console.log("🚀 Detection start result:", result);
    return result;
  }, [detectionControlHandlers]);

  const handleStopDetection = useCallback(async () => {
    const result = await detectionControlHandlers.handleStopDetection();
    console.log("🛑 Detection stop result:", result);
    return result;
  }, [detectionControlHandlers]);

  // Wrapped input handlers
  const handleCameraChange = useCallback((event) => {
    inputHandlers.handleCameraChange(event);
  }, [inputHandlers]);

  const handleTargetLabelChange = useCallback((event) => {
    inputHandlers.handleTargetLabelChange(event);
  }, [inputHandlers]);

  const handleDetectionOptionsChange = useCallback((newOptions) => {
    inputHandlers.handleDetectionOptionsChange(newOptions);
  }, [inputHandlers]);

  // Manual health check button handler
  const handleManualHealthCheck = useCallback(async () => {
    console.log("🩺 Manual health check requested...");
    await performSingleHealthCheck();
  }, [performSingleHealthCheck]);

  // ===== LOT WORKFLOW HANDLERS =====

  // Handle lot selection for workflow
  const handleLotSelection = useCallback(async (lotId) => {
    try {
      console.log('📋 Selecting lot for workflow:', lotId);
      
      const lotResult = await streamManager.getDetectionLot(lotId);
      if (lotResult.success) {
        setCurrentLot(lotResult.lot);
        
        // Set target label from lot's expected piece
        setTargetLabel(lotResult.lot.expected_piece_label || `piece_${lotResult.lot.expected_piece_id}`);
        
        // Load detection history for this lot
        const historyResult = await streamManager.getLotDetectionSessions(lotId);
        if (historyResult.success) {
          setDetectionHistory(historyResult.sessions || []);
        }
        
        showSnackbar(`Lot "${lotResult.lot.lot_name}" selected for workflow`, 'success');
        return { success: true, lot: lotResult.lot };
      } else {
        throw new Error(lotResult.message || 'Failed to load lot');
      }
    } catch (error) {
      console.error('❌ Error selecting lot:', error);
      showSnackbar(`Failed to select lot: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [streamManager, setCurrentLot, setTargetLabel, setDetectionHistory, showSnackbar]);

    // Handle lot workflow detection
  const handleLotWorkflowDetection = useCallback(async () => {
    if (!selectedLotId || !cameraId || !targetLabel) {
      showSnackbar('Lot, camera, and target label are required', 'error');
      return { success: false };
    }

    try {
      setDetectionInProgress(true);
      
      console.log(`🎯 Performing lot workflow detection for lot ${selectedLotId}`);
      
      // FIXED: Use the correct method from BasicStreamManager
      const result = await streamManager.performDetectionWithLotTracking(
        cameraId, 
        targetLabel, 
        {
          lotId: selectedLotId,
          quality: detectionOptions.quality || 85,
          ...detectionOptions
        }
      );

      if (result.success) {
        setLastDetectionResult({
          detected: result.detected,
          confidence: result.confidence,
          processingTime: result.processingTime,
          frameWithOverlay: result.frameWithOverlay,
          streamFrozen: result.streamFrozen,
          timestamp: result.timestamp,
          lotId: result.lotId,
          sessionId: result.sessionId,
          isTargetMatch: result.isTargetMatch,
          detectionRate: result.detectionRate
        });
        
        // Update stream frozen state
        setIsStreamFrozen(result.streamFrozen);
        
        // Update lot state if it's a target match
        if (result.isTargetMatch) {
          setCurrentLot(prev => ({
            ...prev,
            is_target_match: true,
            updated_at: new Date().toISOString()
          }));
        }

        // Reload lot details if loadSelectedLot function is available
        if (loadSelectedLot) {
          await loadSelectedLot();
        }
        
        // Update detection history by fetching lot sessions
        try {
          const historyResult = await streamManager.getLotDetectionSessions(selectedLotId);
          if (historyResult.success) {
            setDetectionHistory(historyResult.sessions || []);
          }
        } catch (historyError) {
          console.warn('Failed to refresh detection history:', historyError);
        }

        // Show appropriate message based on detection result
        const message = result.isTargetMatch 
          ? '🎉 Target match confirmed! Lot requirements met.'
          : result.detected 
            ? '✅ Target detected but verification needed. Continue detecting until correct piece is found.'
            : '❌ Target not found. Please adjust the piece and try again.';

        showSnackbar(message, result.isTargetMatch ? 'success' : 'info');
        
        return {
          success: true,
          detectionResult: {
            detected: result.detected,
            confidence: result.confidence,
            processingTime: result.processingTime,
            streamFrozen: result.streamFrozen,
            isTargetMatch: result.isTargetMatch,
            lotId: result.lotId,
            sessionId: result.sessionId
          },
          lotCompleted: result.isTargetMatch,
          sessionId: result.sessionId
        };
      } else {
        throw new Error(result.message || 'Detection failed');
      }
    } catch (error) {
      console.error('❌ Error in lot workflow detection:', error);
      showSnackbar(`Detection failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    } finally {
      setDetectionInProgress(false);
    }
  }, [
    selectedLotId, 
    cameraId, 
    targetLabel, 
    detectionOptions, 
    streamManager,
    setDetectionInProgress,
    setLastDetectionResult,
    setIsStreamFrozen,
    setCurrentLot,
    setDetectionHistory,
    loadSelectedLot,
    showSnackbar
  ]);

  // Handle starting lot workflow
  const handleStartLotWorkflow = useCallback(async (lotId) => {
    if (!lotId) {
      showSnackbar('Please select a lot first', 'error');
      return { success: false };
    }

    if (!cameraId) {
      showSnackbar('Please select a camera first', 'error');
      return { success: false };
    }

    try {
      console.log('🚀 Starting lot workflow for lot:', lotId);
      
      // Load the lot first
      const lotResult = await handleLotSelection(lotId);
      if (!lotResult.success) {
        return lotResult;
      }

      // Start detection in lot workflow mode
      const startResult = await handleStartDetection();
      
      if (startResult.success) {
        showSnackbar(`Lot workflow started for "${lotResult.lot.lot_name}"`, 'success');
        return { success: true, lot: lotResult.lot };
      } else {
        return startResult;
      }
    } catch (error) {
      console.error('❌ Error starting lot workflow:', error);
      showSnackbar(`Failed to start lot workflow: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [cameraId, handleLotSelection, handleStartDetection, showSnackbar]);

  // Handle stopping lot workflow
  const handleStopLotWorkflow = useCallback(async () => {
    try {
      console.log('🛑 Stopping lot workflow');
      
      // Stop detection first
      await handleStopDetection();
      
      // Clear lot workflow state
      setCurrentLot(null);
      setDetectionHistory([]);
      setLastDetectionResult(null);
      
      showSnackbar('Lot workflow stopped', 'info');
      return { success: true };
    } catch (error) {
      console.error('❌ Error stopping lot workflow:', error);
      showSnackbar(`Failed to stop lot workflow: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [handleStopDetection, setCurrentLot, setDetectionHistory, setLastDetectionResult, showSnackbar]);

  // Handle lot completion verification
  const handleVerifyLotCompletion = useCallback(async (lotId) => {
    try {
      console.log('✅ Verifying lot completion:', lotId);
      
      const result = await streamManager.verifyLotCompletion(lotId);
      
      if (result.success) {
        if (result.isComplete) {
          setCurrentLot(prev => ({
            ...prev,
            is_target_match: true,
            completed_at: result.completedAt
          }));
          showSnackbar('✅ Lot verification completed successfully!', 'success');
        } else {
          showSnackbar('⚠️ Lot requires additional verification', 'warning');
        }
        
        return { success: true, isComplete: result.isComplete };
      } else {
        throw new Error(result.message || 'Verification failed');
      }
    } catch (error) {
      console.error('❌ Error verifying lot completion:', error);
      showSnackbar(`Verification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [streamManager, setCurrentLot, showSnackbar]);

  // ===== HISTORY AND SESSION MANAGEMENT =====

  // Handle refreshing detection history
  const handleRefreshDetectionHistory = useCallback(async (lotId) => {
    try {
      if (!lotId) {
        console.warn('No lot ID provided for history refresh');
        return { success: false };
      }

      console.log('🔄 Refreshing detection history for lot:', lotId);
      
      const historyResult = await streamManager.getLotDetectionSessions(lotId);
      if (historyResult.success) {
        setDetectionHistory(historyResult.sessions || []);
        return { success: true, sessions: historyResult.sessions };
      } else {
        throw new Error(historyResult.message || 'Failed to load history');
      }
    } catch (error) {
      console.error('❌ Error refreshing detection history:', error);
      showSnackbar(`Failed to refresh history: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [streamManager, setDetectionHistory, showSnackbar]);

  // Handle deleting a detection session
  const handleDeleteDetectionSession = useCallback(async (sessionId, lotId) => {
    try {
      console.log('🗑️ Deleting detection session:', sessionId);
      
      const result = await streamManager.deleteDetectionSession(sessionId);
      if (result.success) {
        // Refresh history after deletion
        await handleRefreshDetectionHistory(lotId);
        showSnackbar('Detection session deleted', 'success');
        return { success: true };
      } else {
        throw new Error(result.message || 'Failed to delete session');
      }
    } catch (error) {
      console.error('❌ Error deleting detection session:', error);
      showSnackbar(`Failed to delete session: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [streamManager, handleRefreshDetectionHistory, showSnackbar]);

  // ===== ENHANCED EXISTING HANDLERS =====

  // Handle lot form submission with workflow support
  const handleLotFormSubmit = useCallback(async (lotData, handleCreateLotAndDetect, handleDetectWithExistingLot) => {
    if (lotData.type === 'create_and_detect') {
      const result = await handleCreateLotAndDetect(lotData);
      
      // If lot workflow is requested, start workflow
      if (result.success && lotData.startWorkflow) {
        await handleStartLotWorkflow(result.lotCreated.lot_id);
      }
      
      return result;
    } else if (lotData.type === 'existing_lot') {
      const result = await handleDetectWithExistingLot(lotData);
      
      // If lot workflow is requested, start workflow
      if (result.success && lotData.startWorkflow) {
        await handleStartLotWorkflow(lotData.lotId);
      }
      
      return result;
    }
  }, [handleStartLotWorkflow]);

  // Enhanced detection with lot workflow awareness
  const handleDetectionWithLotContext = useCallback(async (options = {}) => {
    if (lotWorkflowActive && selectedLotId) {
      // Use lot workflow detection
      return await handleLotWorkflowDetection();
    } else {
      // Use regular on-demand detection
      return await handleOnDemandDetection(options);
    }
  }, [lotWorkflowActive, selectedLotId, handleLotWorkflowDetection, handleOnDemandDetection]);

  return {
    // ===== LOT WORKFLOW HANDLERS =====
    handleLotSelection,
    handleLotWorkflowDetection,
    handleStartLotWorkflow,
    handleStopLotWorkflow,
    handleVerifyLotCompletion,
    handleDetectionWithLotContext,
    handleRefreshDetectionHistory,
    handleDeleteDetectionSession,
    
    // ===== SYSTEM CONTROL HANDLERS =====
    handleRefreshSystemProfile,
    handleSwitchToBasicMode,
    handleSwitchToOptimizedMode,
    handleEnableAutoMode,
    handleRunPerformanceTest,
    
    // ===== BASIC MODE HANDLERS =====
    handleOnDemandDetection,
    handleFreezeStream,
    handleUnfreezeStream,
    
    // ===== DETECTION CONTROL HANDLERS =====
    handleStartDetection,
    handleStopDetection,
    
    // ===== INPUT HANDLERS =====
    handleCameraChange,
    handleTargetLabelChange,
    handleDetectionOptionsChange,
    
    // ===== ENHANCED FORM HANDLERS =====
    handleLotFormSubmit,
    handleManualHealthCheck
  };
};

// Custom hook for lot-related handlers (enhanced version)
export const useLotHandlers = ({
  cameraId,
  targetLabel,
  detectionOptions,
  showSnackbar,
  setCurrentLot,
  setLastDetectionResult,
  setIsStreamFrozen,
  fetchExistingLots,
  existingLots,
  streamManager,
  // New lot workflow parameters
  setDetectionHistory,
  setLotWorkflowActive,
  selectedLotId,
  setSelectedLotId
}) => {
  
  // Enhanced create lot and detect function with workflow support
  const handleCreateLotAndDetectEnhanced = useCallback(async (lotData) => {
    if (!cameraId || !targetLabel) {
      showSnackbar("Camera ID and target label are required", "error");
      return { success: false };
    }

    try {
      console.log("🚀 Creating lot and performing detection...", lotData);

      // Use streamManager for enhanced lot creation
      const result = await streamManager.createLotAndDetect({
        lot_name: lotData.lotName.trim(),
        expected_piece_id: parseInt(lotData.expectedPieceId),
        expected_piece_number: parseInt(lotData.expectedPieceNumber),
        expected_piece_label: lotData.expectedPieceLabel || `piece_${lotData.expectedPieceId}`,
        camera_id: cameraId,
        target_label: targetLabel,
        detection_options: detectionOptions,
        workflow_enabled: lotData.enableWorkflow || false
      });

      if (result.success) {
        const lotCreated = result.lot_created;
        const detectionResult = result.detection_result;

        setCurrentLot(lotCreated);
        setLastDetectionResult(detectionResult);
        setIsStreamFrozen(detectionResult.stream_frozen);

        // Initialize detection history if workflow is enabled
        if (lotData.enableWorkflow && setDetectionHistory) {
          setDetectionHistory(result.detection_sessions || []);
        }

        showSnackbar(
          `Lot "${lotCreated.lot_name}" created and detection completed! ${detectionResult.detected_target ? '🎯 Target detected!' : 'Target not found'}`,
          detectionResult.detected_target ? 'success' : 'info'
        );

        console.log("✅ Lot created and detection completed:", {
          lot: lotCreated,
          detection: detectionResult
        });

        // Refresh lots list
        await fetchExistingLots();
        return { 
          success: true, 
          lotCreated, 
          detectionResult,
          workflowEnabled: lotData.enableWorkflow
        };
      } else {
        throw new Error(result.message || 'Failed to create lot and detect');
      }
    } catch (error) {
      console.error("❌ Error creating lot and detecting:", error);
      showSnackbar(
        `Failed to create lot and detect: ${error.response?.data?.detail || error.message}`,
        "error"
      );
      return { success: false, error };
    }
  }, [cameraId, targetLabel, detectionOptions, showSnackbar, setCurrentLot, setLastDetectionResult, setIsStreamFrozen, fetchExistingLots, streamManager, setDetectionHistory]);

  // Enhanced detect with existing lot function with workflow support
  const handleDetectWithExistingLotEnhanced = useCallback(async (lotData) => {
    if (!cameraId || !targetLabel) {
      showSnackbar("Camera ID and target label are required", "error");
      return { success: false };
    }

    try {
      console.log("🎯 Performing detection with existing lot...", lotData);

      // Use streamManager for enhanced lot detection
      const result = await streamManager.detectWithExistingLot({
        lot_id: lotData.lotId,
        camera_id: cameraId,
        target_label: targetLabel,
        detection_options: detectionOptions,
        auto_complete_on_match: true,
        workflow_enabled: lotData.enableWorkflow || false
      });

      if (result.success) {
        const detectionResult = result.detection_result;
        const correctionAction = result.correction_action;
        const lotCompleted = result.lot_completed;

        setLastDetectionResult(detectionResult);
        setIsStreamFrozen(detectionResult.stream_frozen);

        // Update current lot info
        if (lotCompleted) {
          const updatedLot = existingLots.find(lot => lot.lot_id === lotData.lotId);
          if (updatedLot) {
            setCurrentLot({ 
              ...updatedLot, 
              is_target_match: true, 
              completed_at: new Date().toISOString() 
            });
          }
        }

        // Update detection history if workflow is enabled
        if (lotData.enableWorkflow && setDetectionHistory) {
          setDetectionHistory(result.detection_sessions || []);
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
          lotCompleted
        });

        // Refresh lots list
        await fetchExistingLots();
        return { 
          success: true, 
          detectionResult, 
          correctionAction, 
          lotCompleted,
          workflowEnabled: lotData.enableWorkflow
        };
      } else {
        throw new Error(result.message || 'Detection failed');
      }
    } catch (error) {
      console.error("❌ Error detecting with existing lot:", error);
      showSnackbar(
        `Detection with lot failed: ${error.response?.data?.detail || error.message}`,
        "error"
      );
      return { success: false, error };
    }
  }, [cameraId, targetLabel, detectionOptions, existingLots, showSnackbar, setLastDetectionResult, setIsStreamFrozen, setCurrentLot, fetchExistingLots, streamManager, setDetectionHistory]);

    // Enhanced lot form submission with workflow support
  // FIXED: Handle lot completion verification using existing methods
  const handleVerifyLotCompletion = useCallback(async (lotId) => {
    try {
      console.log('✅ Verifying lot completion:', lotId);
      
      // FIXED: Use existing getDetectionLot method to check lot status
      const lotResult = await streamManager.getDetectionLot(lotId);
      
      if (lotResult.success) {
        const lot = lotResult.lot;
        const isComplete = lot.is_target_match || false;
        
        if (isComplete) {
          setCurrentLot(prev => ({
            ...prev,
            is_target_match: true,
            completed_at: lot.completed_at || new Date().toISOString()
          }));
          showSnackbar('✅ Lot verification completed successfully!', 'success');
        } else {
          // Get lot sessions to check if there are any successful detections
          const historyResult = await streamManager.getLotDetectionSessions(lotId);
          if (historyResult.success && historyResult.successfulDetections > 0) {
            showSnackbar('⚠️ Lot has successful detections but needs manual verification', 'warning');
          } else {
            showSnackbar('⚠️ Lot requires additional detection attempts', 'warning');
          }
        }
        
        return { success: true, isComplete, lot };
      } else {
        throw new Error(lotResult.message || 'Verification failed');
      }
    } catch (error) {
      console.error('❌ Error verifying lot completion:', error);
      showSnackbar(`Verification failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [streamManager, setCurrentLot, showSnackbar]);

  // FIXED: Handle deleting a detection session - using manual update since no delete method exists
  const handleDeleteDetectionSession = useCallback(async (sessionId, lotId) => {
    try {
      console.log('🗑️ Delete session not available in current API');
      showSnackbar('Session deletion not supported in current version', 'info');
      return { success: false, message: 'Delete functionality not available' };
    } catch (error) {
      console.error('❌ Error deleting detection session:', error);
      showSnackbar(`Failed to delete session: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [showSnackbar]);

  // FIXED: Enhanced lot form submission with workflow support using existing methods
  const handleEnhancedLotFormSubmit = useCallback(async (lotData) => {
    if (lotData.type === 'create_and_detect') {
      const result = await handleCreateLotAndDetectEnhanced(lotData);
      
      // Start workflow if requested
      if (result.success && lotData.enableWorkflow) {
        setSelectedLotId && setSelectedLotId(result.lotCreated.lot_id);
        setLotWorkflowActive && setLotWorkflowActive(true);
      }
      
      return result;
    } else if (lotData.type === 'existing_lot') {
      const result = await handleDetectWithExistingLotEnhanced(lotData);
      
      // Start workflow if requested
      if (result.success && lotData.enableWorkflow) {
        setSelectedLotId && setSelectedLotId(lotData.lotId);
        setLotWorkflowActive && setLotWorkflowActive(true);
      }
      
      return result;
    }
  }, [handleCreateLotAndDetectEnhanced, handleDetectWithExistingLotEnhanced, setSelectedLotId, setLotWorkflowActive]);
  return {
    handleEnhancedLotFormSubmit,
    handleCreateLotAndDetectEnhanced,
    handleDetectWithExistingLotEnhanced
  };
};