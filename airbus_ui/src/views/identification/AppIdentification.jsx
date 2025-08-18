// AppIdentification.jsx - Main identification component similar to AppDetection
import React, { useState, useEffect, useCallback, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { 
  Box, 
  Grid, 
  Stack, 
  Alert, 
  CircularProgress, 
  Button,
  Snackbar
} from '@mui/material';

// Components
import IdentificationControls from "./components/IdentificationControls";
import IdentificationVideoFeed from "./components/IdentificationVideoFeed";
import IdentificationInfoPanel from "./components/IdentificationInfoPanel";
import BasicModeIdentificationControls from "./components/BasicModeIdentificationControls";

// Services
import { identificationService } from "./service/MainIdentificationService";
import { cameraService } from "../captureImage/CameraService";

// Custom hooks
import { 
  useIdentificationManagement, 
  useIdentificationSystem, 
  useCameraManagement 
} from "./hooks/AppIdentificationHooks";
import { 
  useIdentificationHandlers, 
} from "./handlers/AppIdentificationHandlers";

// Utils
import { 
  initializeIdentificationSystem,
  createRetryInitialization,
  getStateInfo,
  createCleanupFunction,
  IdentificationStates
} from "./utils/AppIdentificationUtils";

export default function AppIdentification() {  
  // UI State
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  // Identification Options
  const [identificationOptions, setIdentificationOptions] = useState({
    streamQuality: 85,
    priority: 1,
    enableAdaptiveQuality: true,
    enableFrameSkipping: true,
    quality: 85
  });

  // Enhanced identification state
  const [identificationHistory, setIdentificationHistory] = useState([]);
  const [isSystemLoading, setIsSystemLoading] = useState(false);

  // Piece types and labels
  const [availablePieceTypes, setAvailablePieceTypes] = useState([]);
  const [pieceLabels, setPieceLabels] = useState(new Map());
  const [loadingLabels, setLoadingLabels] = useState(new Set());

  // Success tracking for stats refresh
  const [lastSuccessfulIdentification, setLastSuccessfulIdentification] = useState(null);

  // Custom hooks for state management
  const identificationManagement = useIdentificationManagement();
  const identificationSystem = useIdentificationSystem();
  const cameraManagement = useCameraManagement();

  // Function to fetch piece label by ID
  const fetchPieceLabel = useCallback(async (pieceId) => {
    if (pieceLabels.has(pieceId) || loadingLabels.has(pieceId)) {
      return pieceLabels.get(pieceId);
    }

    setLoadingLabels(prev => new Set(prev).add(pieceId));

    try {
      const response = await fetch(`/api/artifact_keeper/captureImage/piece_label_byid/${pieceId}`);
      
      if (response.ok) {
        const label = await response.text();
        const cleanLabel = label.replace(/^"|"$/g, '');
        
        setPieceLabels(prev => new Map(prev).set(pieceId, cleanLabel));
        setLoadingLabels(prev => {
          const newSet = new Set(prev);
          newSet.delete(pieceId);
          return newSet;
        });
        
        return cleanLabel;
      } else {
        throw new Error(`Failed to fetch piece label for ID ${pieceId}`);
      }
    } catch (error) {
      console.error(`Error fetching piece label for ID ${pieceId}:`, error);
      setLoadingLabels(prev => {
        const newSet = new Set(prev);
        newSet.delete(pieceId);
        return newSet;
      });
      
      const fallback = `Piece ${pieceId}`;
      setPieceLabels(prev => new Map(prev).set(pieceId, fallback));
      return fallback;
    }
  }, [pieceLabels, loadingLabels]);


  // Load available piece types
  const loadAvailablePieceTypes = useCallback(async () => {
    try {
      console.log('📋 Loading available piece types...');
      const result = await identificationService.getAvailablePieceTypes();
      
      if (result.success) {
        setAvailablePieceTypes(result.availablePieceTypes || []);
        console.log(`✅ Loaded ${result.availablePieceTypes?.length || 0} piece types`);
      } else {
        console.warn('📋 Failed to load piece types:', result.message);
        setAvailablePieceTypes([]);
      }
    } catch (error) {
      console.error('❌ Error loading piece types:', error);
      identificationManagement.showSnackbar(`Failed to load piece types: ${error.message}`, 'error');
      setAvailablePieceTypes([]);
    }
  }, [identificationManagement]);

  // Trigger stats refresh on successful identification
  const triggerStatsRefreshOnSuccess = useCallback(() => {
    console.log('📊 Identification successful - triggering stats refresh');
    setLastSuccessfulIdentification(Date.now());
  }, []);

  // Enhanced identification handlers with proper context
  const identificationHandlers = useIdentificationHandlers({
    cameraId: cameraManagement.cameraId,
    confidenceThreshold,
    identificationInProgress: identificationSystem.identificationInProgress,
    identificationOptions,
    identificationState: identificationSystem.identificationState,
    systemHealth: identificationSystem.systemHealth,
    cameras: cameraManagement.cameras,
    lastHealthCheck: identificationSystem.lastHealthCheck,
    identificationHistory,
    setIsProfileRefreshing: identificationSystem.setIsProfileRefreshing,
    setIdentificationInProgress: identificationSystem.setIdentificationInProgress,
    setLastIdentificationResult: identificationSystem.setLastIdentificationResult,
    setIsStreamFrozen: identificationSystem.setIsStreamFrozen,
    setSelectedCameraId: cameraManagement.setSelectedCameraId,
    setCameraId: cameraManagement.setCameraId,
    setConfidenceThreshold,
    setIdentificationOptions,
    setIdentificationHistory,
    showSnackbar: identificationManagement.showSnackbar,
    performSingleHealthCheck: identificationSystem.performSingleHealthCheck,
    performPostShutdownHealthCheck: identificationSystem.performPostShutdownHealthCheck,
    loadAvailablePieceTypes,
    triggerStatsRefreshOnSuccess
  });

  // Enhanced camera detection
  const handleDetectCameras = useCallback(async () => {
    await cameraManagement.handleDetectCameras(identificationManagement.showSnackbar);
    
    if (cameraManagement.selectedCameraId && 
        !cameraManagement.cameras.some(cam => cam.id.toString() === cameraManagement.selectedCameraId.toString())) {
      identificationSystem.setLastIdentificationResult(null);
      identificationSystem.setIsStreamFrozen(false);
    }
  }, [cameraManagement, identificationManagement, identificationSystem]);

  // System initialization effect
  useEffect(() => {
    const initialize = async () => {
      if (identificationSystem.identificationState === IdentificationStates.INITIALIZING && 
          !identificationSystem.initializationAttempted.current) {
        
        await initializeIdentificationSystem(
          identificationSystem.initializationAttempted,
          identificationSystem.setInitializationError,
          identificationSystem.performInitialHealthCheck,
          loadAvailablePieceTypes
        );
      }
    };

    initialize();
    
    // Cleanup function - no stopMonitoring call needed
    return () => {
      // Any cleanup that needs to happen on unmount
      console.log('🧹 AppIdentification component unmounting...');
    };
  }, [identificationSystem.identificationState, loadAvailablePieceTypes]);

  // Enhanced cleanup
  useEffect(() => {
    const handleBeforeUnload = createCleanupFunction(
      identificationSystem.cleanupRef,
      identificationSystem.identificationState,
      cameraService
    );
    
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      handleBeforeUnload();
    };
  }, [identificationSystem.identificationState, identificationSystem.cleanupRef]);

  // Health check state transitions
  useEffect(() => {
    const handleStateTransition = async () => {
      if (identificationSystem.identificationState === IdentificationStates.READY) {
        const serviceStatus = identificationService.getDetailedStatus();
        
        if (!identificationSystem.healthCheckPerformed.current.postShutdown && 
            !serviceStatus.hasPerformedPostShutdownCheck) {
          console.log("🩺 Triggering post-shutdown health check...");
          await identificationSystem.performPostShutdownHealthCheck();
        } else if (!identificationSystem.healthCheckPerformed.current.initial && 
                   !serviceStatus.hasPerformedInitialHealthCheck) {
          console.log("🩺 Triggering initial health check...");
          await identificationSystem.performInitialHealthCheck();
        }
      }
    };

    handleStateTransition();    
  }, [identificationSystem.identificationState]);

  const retryInitialization = createRetryInitialization(
    identificationSystem.initializationAttempted,
    identificationSystem.setInitializationError,
    identificationSystem.healthCheckPerformed
  );
  
  const stateInfo = getStateInfo(identificationSystem.identificationState);
  const isBasicMode = true; // Identification is always in basic mode

  // Debug logging
  useEffect(() => {
    console.log('🔍 Identification component state debug:', {
      identificationState: identificationSystem.identificationState,
      confidenceThreshold,
      isSystemLoading,
      availablePieceTypesCount: availablePieceTypes.length,
      isStreamFrozen: identificationSystem.isStreamFrozen,
      lastIdentificationResult: identificationSystem.lastIdentificationResult
    });
  }, [
    identificationSystem.identificationState, 
    confidenceThreshold, 
    isSystemLoading, 
    availablePieceTypes.length,
    identificationSystem.isStreamFrozen,
    identificationSystem.lastIdentificationResult
  ]);

return (
  <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
    {/* Info Panel - Fixed prop name */}
    <IdentificationInfoPanel
      isBasicMode={isBasicMode}
      identificationState={identificationSystem.identificationState}
      systemProfile={identificationSystem.systemProfile}
      isProfileRefreshing={identificationSystem.isProfileRefreshing}
      availablePieceTypes={availablePieceTypes}
      confidenceThreshold={confidenceThreshold}
      onRefreshSystemProfile={identificationHandlers.handleRefreshSystemProfile}
      onRunPerformanceTest={identificationHandlers.handleRunPerformanceTest}
      onUpdateConfidenceThreshold={identificationHandlers.handleUpdateConfidenceThreshold}
      IdentificationStates={IdentificationStates}
    />

    {/* System Status Alerts for initialization/shutdown */}
    {identificationSystem.identificationState === IdentificationStates.INITIALIZING && (
      <Alert 
        severity="info" 
        sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
        icon={<CircularProgress size={20} />}
      >
        Initializing identification system... Analyzing system capabilities and loading piece types.
      </Alert>
    )}

    {identificationSystem.identificationState === IdentificationStates.SHUTTING_DOWN && (
      <Alert 
        severity="warning" 
        sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
        icon={<CircularProgress size={20} />}
      >
        System is shutting down... Please wait.
      </Alert>
    )}

    {/* Error and Health Alerts */}
    {identificationSystem.initializationError && (
      <Alert severity="error" sx={{ mb: 2 }}>
        Identification system initialization failed: {identificationSystem.initializationError}
        <Box sx={{ mt: 1 }}>
          <Button 
            variant="outlined" 
            size="small" 
            onClick={retryInitialization}
            disabled={identificationSystem.identificationState === IdentificationStates.INITIALIZING || 
                     identificationSystem.identificationState === IdentificationStates.SHUTTING_DOWN}
          >
            {identificationSystem.identificationState === IdentificationStates.INITIALIZING ? 'Initializing...' : 'Retry Initialization'}
          </Button>
        </Box>
        <br />
        <small>If the issue persists, try refreshing the page or contact support.</small>
      </Alert>
    )}
    
    {!identificationSystem.systemHealth.overall && 
     identificationSystem.identificationState === IdentificationStates.READY && 
     !identificationSystem.initializationError && (
      <Alert severity="warning" sx={{ mb: 2 }}>
        System health check indicates issues. Identification may not work optimally.
        <br />
        <small>
          Streaming: {identificationSystem.systemHealth.streaming?.status || 'Unknown'} | 
          Identification: {identificationSystem.systemHealth.identification?.status || 'Unknown'} | 
          Last checked: {identificationSystem.getHealthCheckAge()}
        </small>
        <Box sx={{ mt: 1 }}>
          <Button 
            variant="outlined" 
            size="small" 
            onClick={identificationHandlers.handleManualHealthCheck}
            disabled={identificationSystem.identificationState !== IdentificationStates.READY}
          >
            Check Health Now
          </Button>
        </Box>
      </Alert>
    )}

    <Grid container spacing={2} columns={12} sx={{ mb: 2 }}>
      {/* Main Content */}
      <Grid size={{ xs: 12, md: 9 }}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            minHeight: { xs: 'auto', md: '500px' }
          }}
        >
          <Stack spacing={3}>
            <IdentificationControls
              selectedCameraId={cameraManagement.selectedCameraId}
              onCameraChange={identificationHandlers.handleCameraChange}
              cameras={cameraManagement.cameras}
              onDetectCameras={handleDetectCameras}
              isDetecting={cameraManagement.isDetecting}
              isSystemReady={stateInfo.canOperate && identificationSystem.identificationState === IdentificationStates.READY}
              systemHealth={identificationSystem.systemHealth}
              identificationOptions={identificationOptions}
              onIdentificationOptionsChange={identificationHandlers.handleIdentificationOptionsChange}
              identificationState={identificationSystem.identificationState}
              confidenceThreshold={confidenceThreshold}
              onConfidenceThresholdChange={identificationHandlers.handleUpdateConfidenceThreshold}
            />
            
            <IdentificationVideoFeed
              isIdentificationActive={identificationSystem.identificationState === IdentificationStates.RUNNING}
              onStartIdentification={identificationHandlers.handleStartIdentification}
              onStopIdentification={identificationHandlers.handleStopIdentification}
              cameraId={cameraManagement.cameraId}
              confidenceThreshold={confidenceThreshold}
              isSystemReady={stateInfo.canOperate}
              identificationOptions={identificationOptions}
              identificationState={identificationSystem.identificationState}
            />
          </Stack>
        </Box>
      </Grid>

      {/* Right Sidebar - Basic Mode Controls */}
      <Grid size={{ xs: 12, md: 3 }}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-start',
            alignItems: 'stretch',
            minHeight: { xs: 'auto', md: '500px' },
            py: 2,
            position: 'relative',
            zIndex: 'auto'
          }}
        >
          {/* Basic Mode Controls - Only show when system is running */}
          {identificationSystem.identificationState === IdentificationStates.RUNNING && 
           cameraManagement.cameraId && (
            <BasicModeIdentificationControls
              isStreamFrozen={identificationSystem.isStreamFrozen}
              identificationInProgress={identificationSystem.identificationInProgress}
              lastIdentificationResult={identificationSystem.lastIdentificationResult}
              confidenceThreshold={confidenceThreshold}
              onPieceIdentification={identificationHandlers.handlePieceIdentification}
              onQuickAnalysis={identificationHandlers.handleQuickAnalysis}
              onFreezeStream={identificationHandlers.handleFreezeStream}
              onUnfreezeStream={identificationHandlers.handleUnfreezeStream}
            />
          )}
        </Box>
      </Grid>
    </Grid>

    {/* Snackbar for notifications */}
    <Snackbar
      open={identificationManagement.snackbar.open}
      autoHideDuration={6000}
      onClose={() => identificationManagement.setSnackbar(prev => ({ ...prev, open: false }))}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
    >
      <Alert
        onClose={() => identificationManagement.setSnackbar(prev => ({ ...prev, open: false }))}
        severity={identificationManagement.snackbar.severity}
        variant="filled"
        sx={{ width: '100%' }}
      >
        {identificationManagement.snackbar.message}
      </Alert>
    </Snackbar>
  </Box>
);
}