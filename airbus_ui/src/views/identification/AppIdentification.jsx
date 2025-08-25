// AppIdentification.jsx - Fixed Group Management State Passing
import React, { useState, useEffect, useCallback, useRef } from "react";
import { 
  Box, 
  Grid, 
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
  
  // Group Management State - FIXED: Use consistent naming
  const [targetGroupName, setTargetGroupName] = useState("");
  const [availableGroups, setAvailableGroups] = useState([]);
  const [currentGroup, setCurrentGroup] = useState(null);
  const [isGroupLoaded, setIsGroupLoaded] = useState(false);
  const [isGroupLoading, setIsGroupLoading] = useState(false);

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

  // Success tracking for stats refresh
  const [lastSuccessfulIdentification, setLastSuccessfulIdentification] = useState(null);

  // Add initialization state tracking to prevent loops
  const [isInitialized, setIsInitialized] = useState(false);
  const initializationInProgress = useRef(false);
  const pieceTypesLoaded = useRef(false);

  // Custom hooks for state management
  const identificationManagement = useIdentificationManagement();
  const identificationSystem = useIdentificationSystem();
  const cameraManagement = useCameraManagement();

  // Group Management Functions
  const loadAvailableGroups = useCallback(async () => {
    try {
      console.log('📋 Loading available groups...');
      const result = await identificationService.getAvailableGroups();
      
      if (result.success) {
        setAvailableGroups(result.available_groups || []);
        setCurrentGroup(result.current_group);
        console.log(`✅ Loaded ${result.available_groups?.length || 0} groups`);
      } else {
        console.warn('📋 Failed to load groups:', result.message);
        setAvailableGroups([]);
      }
    } catch (error) {
      console.error('❌ Error loading groups:', error);
      identificationManagement.showSnackbar(`Failed to load groups: ${error.message}`, 'error');
      setAvailableGroups([]);
    }
  }, [identificationManagement]);

  const handleGroupNameChange = useCallback((groupName) => {
    setTargetGroupName(groupName);
  }, []);

  const handleSelectGroup = useCallback(async (groupName) => {
    if (!groupName?.trim()) {
      identificationManagement.showSnackbar('Please enter a group name', 'warning');
      return;
    }

    setIsGroupLoading(true);
    try {
      console.log(`🔄 Selecting group: ${groupName}`);
      
      // Use the new group selection endpoint
      const response = await fetch(`/api/detection/identification/groups/select/${groupName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();
      
      if (result.success) {
        setCurrentGroup(result.current_group);
        setIsGroupLoaded(result.is_group_loaded);
        setTargetGroupName(result.current_group); // FIXED: Update targetGroupName too
        identificationManagement.showSnackbar(
          `Group "${result.current_group}" loaded successfully`, 
          'success'
        );
        
        // Load piece types for this group
        await loadAvailablePieceTypesForGroup(groupName);
      } else {
        throw new Error(result.detail || result.message || 'Failed to select group');
      }
    } catch (error) {
      console.error('❌ Error selecting group:', error);
      identificationManagement.showSnackbar(
        `Failed to select group "${groupName}": ${error.message}`, 
        'error'
      );
      setIsGroupLoaded(false);
      setCurrentGroup(null);
    } finally {
      setIsGroupLoading(false);
    }
  }, [identificationManagement]);

  // Load available piece types for a specific group
  const loadAvailablePieceTypesForGroup = useCallback(async (groupName) => {
    try {
      console.log(`📋 Loading piece types for group: ${groupName}`);
      
      const response = await fetch(`/api/detection/identification/groups/${groupName}/piece-types`);
      const result = await response.json();
      
      if (result.success) {
        setAvailablePieceTypes(result.available_piece_types || []);
        console.log(`✅ Loaded ${result.available_piece_types?.length || 0} piece types for group ${groupName}`);
      } else {
        console.warn('📋 Failed to load piece types:', result.message);
        setAvailablePieceTypes([]);
      }
    } catch (error) {
      console.error('❌ Error loading piece types:', error);
      setAvailablePieceTypes([]);
    }
  }, []);

  // Updated piece identification handler to include group
  const handlePieceIdentificationWithGroup = useCallback(async (options = {}) => {
    if (!currentGroup && !targetGroupName) {
      identificationManagement.showSnackbar('Please select a group first', 'warning');
      return;
    }

    if (!cameraManagement.cameraId) {
      identificationManagement.showSnackbar('Please select a camera first', 'warning');
      return;
    }

    // FIXED: Use either currentGroup or targetGroupName
    const groupToUse = currentGroup || targetGroupName;

    try {
      identificationSystem.setIdentificationInProgress(true);
      console.log(`🔍 Starting piece identification for group: ${groupToUse}, camera: ${cameraManagement.cameraId}`);
      
      const response = await fetch(`/api/detection/identification/identify/${cameraManagement.cameraId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          group_name: groupToUse,
          freeze_stream: options.freezeStream ?? true,
          quality: identificationOptions.quality || 85
        })
      });

      const result = await response.json();
      
      if (result.success) {
        identificationSystem.setLastIdentificationResult(result);
        identificationSystem.setIsStreamFrozen(result.stream_frozen);
        identificationManagement.showSnackbar(result.message, 'success');
        
        // Update history
        setIdentificationHistory(prev => [{
          timestamp: result.timestamp,
          group_name: result.group_name,
          camera_id: result.camera_id,
          pieces_found: result.summary.total_pieces,
          result: result
        }, ...prev.slice(0, 9)]);
        
        return result;
      } else {
        throw new Error(result.detail || result.message || 'Identification failed');
      }
    } catch (error) {
      console.error('❌ Error in piece identification:', error);
      identificationManagement.showSnackbar(`Identification failed: ${error.message}`, 'error');
      throw error;
    } finally {
      identificationSystem.setIdentificationInProgress(false);
    }
  }, [currentGroup, targetGroupName, cameraManagement.cameraId, identificationOptions, identificationSystem, identificationManagement]);

  // Trigger stats refresh on successful identification
  const triggerStatsRefreshOnSuccess = useCallback(() => {
    console.log('📊 Identification successful - triggering stats refresh');
    setLastSuccessfulIdentification(Date.now());
  }, []);

  // Enhanced identification handlers with group context - FIXED: Pass correct state
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
    
    // FIXED: Pass group state correctly
    targetGroupName, // Use targetGroupName instead of currentGroup
    availableGroups,
    currentGroup,
    
    setIsProfileRefreshing: identificationSystem.setIsProfileRefreshing,
    setIdentificationInProgress: identificationSystem.setIdentificationInProgress,
    setLastIdentificationResult: identificationSystem.setLastIdentificationResult,
    setIsStreamFrozen: identificationSystem.setIsStreamFrozen,
    setSelectedCameraId: cameraManagement.setSelectedCameraId,
    setCameraId: cameraManagement.setCameraId,
    setConfidenceThreshold,
    setIdentificationOptions,
    setIdentificationHistory,
    
    // FIXED: Add missing group state setters
    setTargetGroupName,
    setAvailableGroups,
    setCurrentGroup,
    setIsGroupLoaded,
    setAvailablePieceTypes,
    
    showSnackbar: identificationManagement.showSnackbar,
    performSingleHealthCheck: identificationSystem.performSingleHealthCheck,
    performPostShutdownHealthCheck: identificationSystem.performPostShutdownHealthCheck,
    loadAvailablePieceTypes: () => loadAvailablePieceTypesForGroup(currentGroup || targetGroupName),
    triggerStatsRefreshOnSuccess,
    
    // Override piece identification with group-aware version
    handlePieceIdentification: handlePieceIdentificationWithGroup
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

  // Fixed system initialization effect - prevent endless loops
  useEffect(() => {
    const initialize = async () => {
      // Prevent multiple initialization attempts
      if (initializationInProgress.current || isInitialized) {
        console.log('🔄 Initialization already in progress or completed, skipping...');
        return;
      }

      // Only initialize if state is INITIALIZING and we haven't attempted yet
      if (identificationSystem.identificationState === IdentificationStates.INITIALIZING && 
          !identificationSystem.initializationAttempted.current) {
        
        console.log('🚀 Starting identification system initialization...');
        initializationInProgress.current = true;
        
        try {
          await initializeIdentificationSystem(
            identificationSystem.initializationAttempted,
            identificationSystem.setInitializationError,
            identificationSystem.performInitialHealthCheck,
            loadAvailableGroups // Load groups instead of piece types
          );
          
          setIsInitialized(true);
          console.log('✅ Identification system initialization completed');
        } catch (error) {
          console.error('❌ Identification system initialization failed:', error);
        } finally {
          initializationInProgress.current = false;
        }
      }
    };

    initialize();
    
    // Cleanup function
    return () => {
      console.log('🧹 AppIdentification component unmounting...');
    };
  }, []); // Remove dependencies to prevent re-runs

  // Separate effect for state changes that should trigger re-initialization
  useEffect(() => {
    // Reset initialization flags when explicitly transitioning to INITIALIZING from another state
    if (identificationSystem.identificationState === IdentificationStates.INITIALIZING && 
        isInitialized && 
        !initializationInProgress.current) {
      console.log('🔄 System transitioning back to INITIALIZING state - resetting flags');
      setIsInitialized(false);
      pieceTypesLoaded.current = false;
      identificationSystem.initializationAttempted.current = false;
    }
  }, [identificationSystem.identificationState, isInitialized]);

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

  // Health check state transitions - prevent unnecessary health checks
  useEffect(() => {
    const handleStateTransition = async () => {
      // Only perform health checks when READY and not already in progress
      if (identificationSystem.identificationState === IdentificationStates.READY && 
          !initializationInProgress.current) {
        
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

    // Add a small delay to prevent rapid-fire health checks
    const timeoutId = setTimeout(handleStateTransition, 100);
    
    return () => clearTimeout(timeoutId);    
  }, [identificationSystem.identificationState]);

  const retryInitialization = createRetryInitialization(
    identificationSystem.initializationAttempted,
    identificationSystem.setInitializationError,
    identificationSystem.healthCheckPerformed,
    // Add callback to reset our local flags
    () => {
      setIsInitialized(false);
      pieceTypesLoaded.current = false;
      initializationInProgress.current = false;
    }
  );
  
  const stateInfo = getStateInfo(identificationSystem.identificationState);
  const isBasicMode = true; // Identification is always in basic mode

  // FIXED: Check readiness with proper group validation
  const isReadyForIdentification = stateInfo.canOperate && 
                                  identificationSystem.identificationState === IdentificationStates.READY &&
                                  (isGroupLoaded || targetGroupName) && 
                                  (currentGroup || targetGroupName) && 
                                  cameraManagement.cameraId;

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Info Panel */}
      <IdentificationInfoPanel
        isBasicMode={isBasicMode}
        identificationState={identificationSystem.identificationState}
        systemProfile={identificationSystem.systemProfile}
        isProfileRefreshing={identificationSystem.isProfileRefreshing}
        availablePieceTypes={availablePieceTypes}
        confidenceThreshold={confidenceThreshold}
        currentGroup={currentGroup || targetGroupName}
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
          {initializationInProgress.current ? 
            'Initializing identification system... Please wait.' : 
            'Preparing identification system...'
          }
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
                       identificationSystem.identificationState === IdentificationStates.SHUTTING_DOWN ||
                       initializationInProgress.current}
            >
              {(identificationSystem.identificationState === IdentificationStates.INITIALIZING || 
                initializationInProgress.current) ? 'Initializing...' : 'Retry Initialization'}
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

      {/* Camera Controls - Always visible at top with Group Selection */}
      <Box sx={{ mb: 2 }}>
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
          // FIXED: Pass correct group props
          selectedGroupName={targetGroupName}
          onGroupNameChange={handleGroupNameChange}
          availableGroups={availableGroups}
          onLoadAvailableGroups={loadAvailableGroups}
          isGroupLoaded={isGroupLoaded}
          currentGroup={currentGroup}
          onSelectGroup={handleSelectGroup}
          isGroupLoading={isGroupLoading}
        />
      </Box>

    {/* Main Content Grid - Video Feed and Controls side by side */}
    <Grid container spacing={3} columns={12}>
      {/* Video Feed Column */}
      <Grid size={{ xs: 12, md: 8 }}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: { xs: 'auto', md: '500px' }
          }}
        >
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
        </Box>
      </Grid>

      {/* Basic Controls Column - Always visible */}
      <Grid size={{ xs: 12, md: 4 }}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'stretch',
            minHeight: { xs: 'auto', md: '500px' }
          }}
        >
          <BasicModeIdentificationControls
            isStreamFrozen={identificationSystem.isStreamFrozen}
            identificationInProgress={identificationSystem.identificationInProgress}
            lastIdentificationResult={identificationSystem.lastIdentificationResult}
            confidenceThreshold={confidenceThreshold}
            onPieceIdentification={identificationHandlers.handlePieceIdentification}
            onFreezeStream={identificationHandlers.handleFreezeStream}
            onUnfreezeStream={identificationHandlers.handleUnfreezeStream}
          />
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