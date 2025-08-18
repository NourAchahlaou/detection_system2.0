// handlers/AppIdentificationHandlers.js - Simplified identification handlers
import { useCallback } from "react";
import {
  createIdentificationControlHandlers,
  createIdentificationStreamHandlers,
  createInputHandlers
} from "../utils/AppIdentificationUtils";
import { identificationService } from "../service/MainIdentificationService";
// Custom hook for all identification-related handlers
export const useIdentificationHandlers = ({
  // State values
  cameraId,
  identificationInProgress,
  identificationState,
  systemHealth,
  cameras,
  lastHealthCheck,
  confidenceThreshold,
  
  // State setters
  setIdentificationInProgress,
  setLastIdentificationResult,
  setIsStreamFrozen,
  setSelectedCameraId,
  setCameraId,
  setConfidenceThreshold,
  setAvailablePieceTypes,
  
  // Functions
  showSnackbar,
  performSingleHealthCheck,
  performPostShutdownHealthCheck,
  loadAvailablePieceTypes,
  
  // Refs
  lastHealthCheck: lastHealthCheckRef
}) => {

  // ===== IDENTIFICATION HANDLERS =====

  // Create identification control handlers
  const identificationControlHandlers = createIdentificationControlHandlers(
    cameraId,
    identificationState,
    systemHealth,
    cameras,
    showSnackbar,
    performSingleHealthCheck,
    lastHealthCheckRef,
    setIsStreamFrozen,
    setLastIdentificationResult,
    setIdentificationInProgress,
    performPostShutdownHealthCheck
  );

  // Create identification stream handlers
  const identificationStreamHandlers = createIdentificationStreamHandlers(
    cameraId,
    identificationInProgress,
    confidenceThreshold,
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

  // Wrapped identification control handlers
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

  // Wrapped identification stream handlers
  const handlePieceIdentification = useCallback(async (options = {}) => {
    const result = await identificationStreamHandlers.handlePieceIdentification(options);
    return result;
  }, [identificationStreamHandlers]);

  const handleQuickAnalysis = useCallback(async (options = {}) => {
    const result = await identificationStreamHandlers.handleQuickAnalysis(options);
    return result;
  }, [identificationStreamHandlers]);

  const handleFreezeStream = useCallback(async () => {
    await identificationStreamHandlers.handleFreezeStream();
  }, [identificationStreamHandlers]);

  const handleUnfreezeStream = useCallback(async () => {
    await identificationStreamHandlers.handleUnfreezeStream();
  }, [identificationStreamHandlers]);

  // Wrapped input handlers
  const handleCameraChange = useCallback((event) => {
    inputHandlers.handleCameraChange(event);
  }, [inputHandlers]);

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

  return {
    // ===== IDENTIFICATION CONTROL HANDLERS =====
    handleStartIdentification,
    handleStopIdentification,
    
    // ===== IDENTIFICATION STREAM HANDLERS =====
    handlePieceIdentification,
    handleQuickAnalysis,
    handleFreezeStream,
    handleUnfreezeStream,
    
    // ===== INPUT HANDLERS =====
    handleCameraChange,
    
    // ===== SETTINGS HANDLERS =====
    handleConfidenceThresholdChange,
    handleLoadPieceTypes,
    handleGetIdentificationSettings,
    
    // ===== HEALTH CHECK HANDLERS =====
    handleManualHealthCheck
  };
};