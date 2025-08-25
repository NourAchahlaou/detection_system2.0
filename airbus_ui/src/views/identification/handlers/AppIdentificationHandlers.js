// handlers/AppIdentificationHandlers.js - Updated for group-based identification
import { useCallback } from "react";
import {
  createIdentificationControlHandlers,
  createIdentificationStreamHandlers,
  createInputHandlers,
  createGroupManagementHandlers
} from "../utils/AppIdentificationUtils";
import { identificationService } from "../service/MainIdentificationService";

// Custom hook for all identification-related handlers with group support
export const useIdentificationHandlers = ({
  // State values
  cameraId,
  identificationInProgress,
  identificationState,
  systemHealth,
  cameras,
  lastHealthCheck,
  confidenceThreshold,
  targetGroupName,
  availableGroups,
  currentGroup,
  
  // State setters
  setIdentificationInProgress,
  setLastIdentificationResult,
  setIsStreamFrozen,
  setSelectedCameraId,
  setCameraId,
  setConfidenceThreshold,
  setAvailablePieceTypes,
  setTargetGroupName,
  setAvailableGroups,
  setCurrentGroup,
  setIsGroupLoaded,
  
  // Functions
  showSnackbar,
  performSingleHealthCheck,
  performPostShutdownHealthCheck,
  loadAvailablePieceTypes,
  
  // Refs
  lastHealthCheck: lastHealthCheckRef
}) => {

  // ===== GROUP MANAGEMENT HANDLERS =====

  // Create group management handlers
  const groupManagementHandlers = createGroupManagementHandlers(
    targetGroupName,
    currentGroup,
    availableGroups,
    showSnackbar,
    setTargetGroupName,
    setAvailableGroups,
    setCurrentGroup,
    setIsGroupLoaded,
    setAvailablePieceTypes
  );

  // ===== IDENTIFICATION HANDLERS =====

  // Create identification control handlers
  const identificationControlHandlers = createIdentificationControlHandlers(
    cameraId,
    identificationState,
    systemHealth,
    cameras,
    targetGroupName,
    showSnackbar,
    performSingleHealthCheck,
    lastHealthCheckRef,
    setIsStreamFrozen,
    setLastIdentificationResult,
    setIdentificationInProgress,
    performPostShutdownHealthCheck
  );

  // Create identification stream handlers with group support
  const identificationStreamHandlers = createIdentificationStreamHandlers(
    cameraId,
    identificationInProgress,
    confidenceThreshold,
    targetGroupName,
    showSnackbar,
    setIdentificationInProgress,
    setLastIdentificationResult,
    setIsStreamFrozen
  );

  // Create input handlers
  const inputHandlers = createInputHandlers(
    cameras,
    showSnackbar,
    setSelectedCameraId,
    setCameraId,
    setLastIdentificationResult,
    setIsStreamFrozen,
    identificationState
  );

  // ===== GROUP HANDLERS =====

  const handleLoadAvailableGroups = useCallback(async () => {
    return await groupManagementHandlers.handleLoadAvailableGroups();
  }, [groupManagementHandlers]);

  const handleSelectGroup = useCallback(async (groupName) => {
    return await groupManagementHandlers.handleSelectGroup(groupName);
  }, [groupManagementHandlers]);

  const handleTargetGroupNameChange = useCallback((event) => {
    groupManagementHandlers.handleTargetGroupNameChange(event);
  }, [groupManagementHandlers]);

  const handleGetPieceTypesForGroup = useCallback(async (groupName) => {
    return await groupManagementHandlers.handleGetPieceTypesForGroup(groupName);
  }, [groupManagementHandlers]);

  const handleSwitchGroupAndIdentify = useCallback(async (newGroupName, options = {}) => {
    return await groupManagementHandlers.handleSwitchGroupAndIdentify(cameraId, newGroupName, options);
  }, [groupManagementHandlers, cameraId]);

  // ===== IDENTIFICATION CONTROL HANDLERS =====

  const handleStartIdentification = useCallback(async () => {
    const result = await identificationControlHandlers.handleStartIdentification();
    console.log("🚀 Identification start result:", result);
    return result;
  }, [identificationControlHandlers]);

  const handleStopIdentification = useCallback(async () => {
    const result = await identificationControlHandlers.handleStopIdentification();
    console.log("🛑 Identification stop result:", result);
    return result;
  }, [identificationControlHandlers]);

  // ===== IDENTIFICATION STREAM HANDLERS =====

  const handlePieceIdentification = useCallback(async (options = {}) => {
    // Validate group selection first
    if (!targetGroupName || targetGroupName.trim() === '') {
      showSnackbar("Please enter a target group name first.", "error");
      return { success: false, error: "No group name provided" };
    }

    const result = await identificationStreamHandlers.handlePieceIdentification({
      groupName: targetGroupName.trim(),
      ...options
    });
    return result;
  }, [identificationStreamHandlers, targetGroupName, showSnackbar]);

  const handleQuickAnalysis = useCallback(async (options = {}) => {
    // Validate group selection first
    if (!targetGroupName || targetGroupName.trim() === '') {
      showSnackbar("Please enter a target group name first.", "error");
      return { success: false, error: "No group name provided" };
    }

    const result = await identificationStreamHandlers.handleQuickAnalysis({
      groupName: targetGroupName.trim(),
      ...options
    });
    return result;
  }, [identificationStreamHandlers, targetGroupName, showSnackbar]);

  const handleBatchIdentification = useCallback(async (options = {}) => {
    // Validate group selection first
    if (!targetGroupName || targetGroupName.trim() === '') {
      showSnackbar("Please enter a target group name first.", "error");
      return { success: false, error: "No group name provided" };
    }

    const result = await identificationStreamHandlers.handleBatchIdentification({
      groupName: targetGroupName.trim(),
      ...options
    });
    return result;
  }, [identificationStreamHandlers, targetGroupName, showSnackbar]);

  const handleFreezeStream = useCallback(async () => {
    await identificationStreamHandlers.handleFreezeStream();
  }, [identificationStreamHandlers]);

  const handleUnfreezeStream = useCallback(async () => {
    await identificationStreamHandlers.handleUnfreezeStream();
  }, [identificationStreamHandlers]);

  // ===== INPUT HANDLERS =====

  const handleCameraChange = useCallback((event) => {
    inputHandlers.handleCameraChange(event);
  }, [inputHandlers]);

  // ===== SETTINGS HANDLERS =====

  // Manual health check button handler
  const handleManualHealthCheck = useCallback(async () => {
    console.log("🩺 Manual health check requested...");
    await performSingleHealthCheck();
  }, [performSingleHealthCheck]);

  // Confidence threshold handler
  const handleConfidenceThresholdChange = useCallback(async (newThreshold) => {
    try {
      console.log(`🎯 Updating confidence threshold to ${newThreshold}...`);
      const result = await identificationService.updateConfidenceThreshold(newThreshold);
      
      if (result.success) {
        setConfidenceThreshold(result.newThreshold);
        showSnackbar(`Confidence threshold updated to ${result.newThreshold}`, 'success');
        return { success: true };
      } else {
        throw new Error('Failed to update confidence threshold');
      }
    } catch (error) {
      console.error('❌ Error updating confidence threshold:', error);
      showSnackbar(`Failed to update confidence threshold: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [setConfidenceThreshold, showSnackbar]);

  // Update confidence threshold handler (legacy support)
  const handleUpdateConfidenceThreshold = useCallback(async (newThreshold) => {
    return await handleConfidenceThresholdChange(newThreshold);
  }, [handleConfidenceThresholdChange]);

  // Load piece types handler
  const handleLoadPieceTypes = useCallback(async () => {
    try {
      console.log('📋 Loading available piece types...');
      const result = await identificationService.getAvailablePieceTypes();
      
      if (result.success) {
        setAvailablePieceTypes(result.availablePieceTypes || []);
        showSnackbar(`Loaded ${result.totalClasses} piece types`, 'success');
        return { success: true, pieceTypes: result.availablePieceTypes };
      } else {
        throw new Error('Failed to load piece types');
      }
    } catch (error) {
      console.error('❌ Error loading piece types:', error);
      showSnackbar(`Failed to load piece types: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [setAvailablePieceTypes, showSnackbar]);

  // Get identification settings handler
  const handleGetIdentificationSettings = useCallback(async () => {
    try {
      console.log('⚙️ Getting identification settings...');
      const result = await identificationService.getIdentificationSettings();
      
      if (result.success) {
        return { success: true, settings: result.settings };
      } else {
        throw new Error('Failed to get identification settings');
      }
    } catch (error) {
      console.error('❌ Error getting identification settings:', error);
      showSnackbar(`Failed to get identification settings: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [showSnackbar]);

  // Enhanced identification options handler
  const handleIdentificationOptionsChange = useCallback((newOptions) => {
    console.log('🔧 Identification options changed:', newOptions);
    // This would be handled by the parent component
    // Just log for now since options are passed down
  }, []);

  // System profile refresh handler
  const handleRefreshSystemProfile = useCallback(async () => {
    try {
      console.log('🔄 Refreshing system profile...');
      showSnackbar('Refreshing system profile...', 'info');
      
      // Trigger health check which will update system profile
      await performSingleHealthCheck();
      
      // Load available groups
      await handleLoadAvailableGroups();
      
      showSnackbar('System profile refreshed successfully', 'success');
      return { success: true };
    } catch (error) {
      console.error('❌ Error refreshing system profile:', error);
      showSnackbar(`Failed to refresh system profile: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [performSingleHealthCheck, handleLoadAvailableGroups, showSnackbar]);

  // Performance test handler
  const handleRunPerformanceTest = useCallback(async () => {
    try {
      console.log('🚀 Running performance test...');
      showSnackbar('Running performance test...', 'info');
      
      // This would need to be implemented based on your performance test requirements
      // For now, just show a placeholder message
      showSnackbar('Performance test completed (placeholder)', 'success');
      return { success: true };
    } catch (error) {
      console.error('❌ Error running performance test:', error);
      showSnackbar(`Performance test failed: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }, [showSnackbar]);

  return {
    // ===== GROUP MANAGEMENT HANDLERS =====
    handleLoadAvailableGroups,
    handleSelectGroup,
    handleTargetGroupNameChange,
    handleGetPieceTypesForGroup,
    handleSwitchGroupAndIdentify,
    
    // ===== IDENTIFICATION CONTROL HANDLERS =====
    handleStartIdentification,
    handleStopIdentification,
    
    // ===== IDENTIFICATION STREAM HANDLERS =====
    handlePieceIdentification,
    handleQuickAnalysis,
    handleBatchIdentification,
    handleFreezeStream,
    handleUnfreezeStream,
    
    // ===== INPUT HANDLERS =====
    handleCameraChange,
    handleTargetGroupNameChange,
    
    // ===== SETTINGS HANDLERS =====
    handleConfidenceThresholdChange,
    handleUpdateConfidenceThreshold,
    handleLoadPieceTypes,
    handleGetIdentificationSettings,
    handleIdentificationOptionsChange,
    
    // ===== HEALTH CHECK HANDLERS =====
    handleManualHealthCheck,
    
    // ===== SYSTEM HANDLERS =====
    handleRefreshSystemProfile,
    handleRunPerformanceTest
  };
};