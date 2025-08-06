// pages/AppDetection.jsx - Fixed infinite fetch loop
import React, { useState, useEffect, useCallback, useRef } from "react";
import { 
  Box, 
  Grid, 
  Stack, 
  Alert, 
  CircularProgress, 
  Typography,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Snackbar
} from '@mui/material';
import { 
  Refresh, 
  Speed, 
  Computer, 
  Smartphone,
  Inventory,
  CheckCircle,
  Warning,
  Close
} from '@mui/icons-material';

// Components
import DetectionControls from "./components/DetectionControls";
import DetectionVideoFeed from "./components/DetectionVideoFeed";
import DetectionLotForm from "./components/DetectionLotForm";
import SystemPerformancePanel from "./components/SystemPerformancePanel";
import BasicModeControls from "./components/BasicModeControls";

// Services - Use StreamManager instead of direct API calls
import { detectionService } from "./service/DetectionService";
import { cameraService } from "../captureImage/CameraService";

// Custom hooks
import { 
  useLotManagement, 
  useDetectionSystem, 
  useCameraManagement 
} from "./hooks/AppDetectionHooks";
import { 
  useDetectionHandlers, 
  useLotHandlers 
} from "./handlers/AppDetectionHandlers";

// Utils
import { 
  initializeDetectionSystem,
  createRetryInitialization,
  getStateInfo,
  getModeDisplayInfo,
  createCleanupFunction,
  DetectionStates
} from "./utils/AppDetectionUtils";

export default function AppDetection() {
  // UI State
  const [targetLabel, setTargetLabel] = useState("");
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  
  // Detection Options
  const [detectionOptions, setDetectionOptions] = useState({
    detectionFps: 5.0,
    streamQuality: 85,
    priority: 1,
    enableAdaptiveQuality: true,
    enableFrameSkipping: true,
    quality: 85
  });

  // FIXED: More robust caching system with better control
  const lotsCache = useRef({
    data: new Map(),
    lastFetch: 0,
    isValid: false,
    fetchPromise: null
  });
  
  const LOTS_CACHE_DURATION = 30000; // 30 seconds cache
  const FETCH_DEBOUNCE_TIME = 1000; // 1 second debounce

  // Custom hooks for state management
  const lotManagement = useLotManagement();
  const detectionSystem = useDetectionSystem();
  const cameraManagement = useCameraManagement();

  // Get StreamManager instance from detection service
  const streamManager = detectionService.streamManager;

  // FIXED: Stable fetch function with better caching and debouncing
  const fetchExistingLotsEfficient = useCallback(async (forceRefresh = false) => {
    const now = Date.now();
    
    // Check if we have valid cached data
    if (!forceRefresh && 
        lotsCache.current.isValid && 
        lotsCache.current.data.size > 0 && 
        (now - lotsCache.current.lastFetch) < LOTS_CACHE_DURATION) {
      console.log('📋 Using cached lots data (valid cache)');
      const cachedData = Array.from(lotsCache.current.data.values());
      lotManagement.setExistingLots(cachedData);
      return cachedData;
    }

    // Prevent concurrent fetches - return existing promise if one is in progress
    if (lotsCache.current.fetchPromise) {
      console.log('📋 Fetch already in progress, waiting for completion...');
      try {
        return await lotsCache.current.fetchPromise;
      } catch (error) {
        console.warn('📋 Previous fetch failed, will retry');
        lotsCache.current.fetchPromise = null;
      }
    }

    // Create new fetch promise
    lotsCache.current.fetchPromise = (async () => {
      try {
        console.log('📋 Fetching lots via StreamManager...');

        const result = await streamManager.getAllDetectionLots();

        if (result.success && result.lots) {
          // Update cache atomically
          lotsCache.current.data.clear();
          result.lots.forEach(lot => {
            lotsCache.current.data.set(lot.lot_id, lot);
          });
          lotsCache.current.lastFetch = now;
          lotsCache.current.isValid = true;

          // Update state
          lotManagement.setExistingLots(result.lots);
          
          console.log(`✅ Fetched ${result.lots.length} lots via StreamManager`);
          return result.lots;
        } else {
          throw new Error(result.message || 'Failed to fetch lots');
        }

      } catch (error) {
        console.error('❌ Error fetching lots via StreamManager:', error);
        
        // Invalidate cache on error
        lotsCache.current.isValid = false;
        
        // Don't show error message on component unmount or if it's a network issue during shutdown
        if (detectionSystem.detectionState !== DetectionStates.SHUTTING_DOWN) {
          lotManagement.showSnackbar(`Failed to fetch lots: ${error.message}`, 'error');
        }
        
        // Return current state on error
        return lotManagement.existingLots;
      } finally {
        // Clear the promise reference
        lotsCache.current.fetchPromise = null;
      }
    })();

    return lotsCache.current.fetchPromise;
  }, [streamManager, lotManagement, detectionSystem.detectionState]);

  // FIXED: Debounced cache invalidation
  const invalidateLotsCache = useCallback(() => {
    console.log('🗑️ Invalidating lots cache');
    lotsCache.current.isValid = false;
    lotsCache.current.data.clear();
  }, []);

  // Enhanced lot creation using StreamManager
  const handleCreateLotOnly = useCallback(async (lotData) => {
    try {
      console.log('Creating lot with StreamManager:', lotData);
      
      // Validate required fields
      if (!lotData.lotName?.trim()) {
        lotManagement.showSnackbar('Lot name is required', 'error');
        return { success: false, error: 'Lot name is required' };
      }
      
      if (!lotData.expectedPieceLabel?.trim()) {
        lotManagement.showSnackbar('Please select a piece', 'error');
        return { success: false, error: 'Piece selection is required' };
      }
      
      if (!lotData.expectedPieceId) {
        lotManagement.showSnackbar('Piece ID not found', 'error');
        return { success: false, error: 'Piece ID is required' };
      }
      
      if (!lotData.expectedPieceNumber || lotData.expectedPieceNumber <= 0) {
        lotManagement.showSnackbar('Expected piece number must be positive', 'error');
        return { success: false, error: 'Valid piece number is required' };
      }

      lotManagement.setLotOperationInProgress(true);

      // Use StreamManager's createDetectionLot method
      const result = await streamManager.createDetectionLot(
        lotData.lotName.trim(),
        parseInt(lotData.expectedPieceId),
        parseInt(lotData.expectedPieceNumber)
      );

      if (result.success) {
        console.log('Lot created successfully via StreamManager:', result.lotData);

        // Update current lot and invalidate cache
        lotManagement.setCurrentLot(result.lotData);
        invalidateLotsCache();
        
        // Efficiently refresh lots list with a small delay to avoid immediate refetch
        setTimeout(() => {
          fetchExistingLotsEfficient(true);
        }, 500);
        
        lotManagement.showSnackbar(
          `Lot "${lotData.lotName}" created successfully!`, 
          'success'
        );

        return { success: true, lot: result.lotData };
      } else {
        throw new Error(result.message || 'Failed to create lot');
      }

    } catch (error) {
      console.error('Error creating lot via StreamManager:', error);
      const errorMessage = error.message || 'Failed to create lot';
      lotManagement.showSnackbar(`Failed to create lot: ${errorMessage}`, 'error');
      return { success: false, error: errorMessage };
    } finally {
      lotManagement.setLotOperationInProgress(false);
    }
  }, [lotManagement, streamManager, invalidateLotsCache, fetchExistingLotsEfficient]);

  // Enhanced lot creation with detection using StreamManager
  const handleCreateLotAndDetect = useCallback(async (lotData) => {
    try {
      console.log('Creating lot and detecting with StreamManager:', lotData);
      
      // Validate inputs
      if (!cameraManagement.cameraId) {
        lotManagement.showSnackbar('Please select a camera first', 'error');
        return { success: false, error: 'No camera selected' };
      }

      if (!lotData.expectedPieceLabel?.trim()) {
        lotManagement.showSnackbar('Please select a piece for detection', 'error');
        return { success: false, error: 'No target label specified' };
      }

      lotManagement.setLotOperationInProgress(true);

      // Use StreamManager's createLotAndDetect method
      const result = await streamManager.createLotAndDetect(
        cameraManagement.cameraId,
        lotData.lotName.trim(),
        parseInt(lotData.expectedPieceId),
        parseInt(lotData.expectedPieceNumber),
        lotData.expectedPieceLabel || targetLabel,
        {
          quality: detectionOptions.quality || 85
        }
      );

      if (result.success) {
        console.log('Lot created and detection performed via StreamManager:', result);

        // Update states
        lotManagement.setCurrentLot(result.lotCreated);
        detectionSystem.setLastDetectionResult({
          detected: result.detectionResult.detected,
          confidence: result.detectionResult.confidence,
          processingTime: result.detectionResult.processingTime,
          timestamp: result.detectionResult.timestamp,
          frameWithOverlay: result.detectionResult.frameWithOverlay,
          lotId: result.detectionResult.lotId,
          sessionId: result.detectionResult.sessionId,
          isTargetMatch: result.detectionResult.isTargetMatch
        });
        
        detectionSystem.setIsStreamFrozen(result.detectionResult.streamFrozen);
        
        // Invalidate cache and refresh lots with delay
        invalidateLotsCache();
        setTimeout(() => {
          fetchExistingLotsEfficient(true);
        }, 500);

        lotManagement.showSnackbar(
          `Lot "${lotData.lotName}" created and detection completed! ${result.detectionResult.detected ? '✅ Target detected' : '❌ Target not found'}`, 
          result.detectionResult.detected ? 'success' : 'warning'
        );

        return { success: true, lot: result.lotCreated, detectionResult: result.detectionResult };
      } else {
        throw new Error(result.message || 'Failed to create lot and detect');
      }

    } catch (error) {
      console.error('Error in create lot and detect via StreamManager:', error);
      const errorMessage = error.message || 'Failed to create lot and start detection';
      lotManagement.showSnackbar(`Error: ${errorMessage}`, 'error');
      return { success: false, error: errorMessage };
    } finally {
      lotManagement.setLotOperationInProgress(false);
    }
  }, [cameraManagement.cameraId, lotManagement, detectionSystem, streamManager, targetLabel, detectionOptions, invalidateLotsCache, fetchExistingLotsEfficient]);

  // Enhanced lot handlers with StreamManager integration
  const lotHandlers = useLotHandlers({
    cameraId: cameraManagement.cameraId,
    targetLabel,
    detectionOptions,
    showSnackbar: lotManagement.showSnackbar,
    setCurrentLot: lotManagement.setCurrentLot,
    setLastDetectionResult: detectionSystem.setLastDetectionResult,
    setIsStreamFrozen: detectionSystem.setIsStreamFrozen,
    fetchExistingLots: fetchExistingLotsEfficient,
    existingLots: lotManagement.existingLots,
    streamManager // Pass StreamManager for enhanced operations
  });

  // Enhanced detection handlers with StreamManager
  const detectionHandlers = useDetectionHandlers({
    // State values
    cameraId: cameraManagement.cameraId,
    targetLabel,
    currentLot: lotManagement.currentLot,
    detectionInProgress: detectionSystem.detectionInProgress,
    detectionOptions,
    detectionState: detectionSystem.detectionState,
    currentStreamingType: detectionSystem.currentStreamingType,
    systemHealth: detectionSystem.systemHealth,
    cameras: cameraManagement.cameras,
    lastHealthCheck: detectionSystem.lastHealthCheck,
    
    // State setters
    setIsProfileRefreshing: detectionSystem.setIsProfileRefreshing,
    setDetectionInProgress: detectionSystem.setDetectionInProgress,
    setOnDemandDetecting: detectionSystem.setOnDemandDetecting,
    setLastDetectionResult: detectionSystem.setLastDetectionResult,
    setIsStreamFrozen: detectionSystem.setIsStreamFrozen,
    setCurrentLot: lotManagement.setCurrentLot,
    setSelectedCameraId: cameraManagement.setSelectedCameraId,
    setCameraId: cameraManagement.setCameraId,
    setTargetLabel,
    setDetectionOptions,
    
    // Functions
    showSnackbar: lotManagement.showSnackbar,
    fetchExistingLots: fetchExistingLotsEfficient,
    performSingleHealthCheck: detectionSystem.performSingleHealthCheck,
    performPostShutdownHealthCheck: detectionSystem.performPostShutdownHealthCheck,
    
    // Enhanced functions
    streamManager, // Pass StreamManager for detection operations
    
    // Refs
    lastHealthCheck: detectionSystem.lastHealthCheck
  });

  // Enhanced lot form submission handler with StreamManager
const handleLotFormSubmit = useCallback(async (lotData) => {
  console.log('Lot form submitted with data:', lotData);
  
  if (lotData.type === 'existing_lot') {
    // Use existing lot with enhanced detection
    if (lotData.lotId && cameraManagement.cameraId && targetLabel) {
      try {
        lotManagement.setLotOperationInProgress(true);
        
        // Use StreamManager's performDetectionWithLotTracking
        const result = await streamManager.performDetectionWithLotTracking(
          cameraManagement.cameraId,
          targetLabel,
          {
            lotId: parseInt(lotData.lotId),
            quality: detectionOptions.quality || 85
          }
        );

        if (result.success) {
          // Update states
          detectionSystem.setLastDetectionResult({
            detected: result.detected,
            confidence: result.confidence,
            processingTime: result.processingTime,
            timestamp: result.timestamp,
            frameWithOverlay: result.frameWithOverlay,
            lotId: result.lotId,
            sessionId: result.sessionId,
            isTargetMatch: result.isTargetMatch
          });
          
          detectionSystem.setIsStreamFrozen(result.streamFrozen);
          
          // Get updated lot info
          const lotResult = await streamManager.getDetectionLot(result.lotId);
          if (lotResult.success) {
            lotManagement.setCurrentLot(lotResult.lotData);
          }

          // Refresh lots if lot status changed
          if (result.isTargetMatch) {
            invalidateLotsCache();
            setTimeout(() => {
              fetchExistingLotsEfficient(true);
            }, 500);
          }

          lotManagement.showSnackbar(
            `Detection completed! ${result.detected ? '✅ Target detected' : '❌ Target not found'}`, 
            result.detected ? 'success' : 'warning'
          );

          return { success: true, detectionResult: result };
        } else {
          throw new Error(result.message || 'Detection failed');
        }
      } catch (error) {
        console.error('Error in lot-tracked detection:', error);
        lotManagement.showSnackbar(`Detection failed: ${error.message}`, 'error');
        return { success: false, error: error.message };
      } finally {
        lotManagement.setLotOperationInProgress(false);
      }
    } else {
      const result = await lotHandlers.handleEnhancedLotFormSubmit(lotData);
      return result;
    }
  } else if (lotData.type === 'create_only') {
    // Create lot only
    const result = await handleCreateLotOnly(lotData);
    return result;
  } else {
    console.error('Unknown lot operation type:', lotData.type);
    lotManagement.showSnackbar('Unknown operation type', 'error');
    return { success: false, error: 'Unknown operation type' };
  }
}, [lotHandlers, handleCreateLotOnly, lotManagement, 
    cameraManagement.cameraId, targetLabel, detectionOptions, streamManager, detectionSystem, invalidateLotsCache, fetchExistingLotsEfficient]);

  // FIXED: Load existing lots ONLY once when component mounts
  useEffect(() => {
    let mounted = true;
    
    const loadInitialLots = async () => {
      if (mounted && !lotsCache.current.isValid) {
        console.log('📋 Loading initial lots...');
        await fetchExistingLotsEfficient(false);
      }
    };
    
    loadInitialLots();
    
    return () => {
      mounted = false;
    };
  }, []); // Empty dependency array - only run once

  // FIXED: Removed the problematic useEffect that was causing infinite loops
  // Instead, only invalidate cache when lots are actually modified, not on every state change

  // System initialization effect
  useEffect(() => {
    const initialize = async () => {
      if (detectionSystem.detectionState === DetectionStates.INITIALIZING && 
          !detectionSystem.initializationAttempted.current) {
        
        await initializeDetectionSystem(
          detectionSystem.initializationAttempted,
          detectionSystem.setInitializationError,
          detectionSystem.performInitialHealthCheck,
          detectionSystem.startStatsMonitoring
        );
      }
    };

    initialize();
    
    return () => {
      detectionSystem.stopMonitoring();
    };
  }, [detectionSystem.detectionState]);

  // Enhanced cleanup with proper shutdown signaling
  useEffect(() => {
    const handleBeforeUnload = createCleanupFunction(
      detectionSystem.cleanupRef,
      detectionSystem.stopMonitoring,
      detectionSystem.detectionState,
      cameraService
    );
    
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      handleBeforeUnload();
      // Clear cache on unmount
      lotsCache.current.data.clear();
      lotsCache.current.isValid = false;
      if (lotsCache.current.fetchPromise) {
        lotsCache.current.fetchPromise = null;
      }
    };
  }, [detectionSystem.detectionState, detectionSystem.stopMonitoring, detectionSystem.cleanupRef]);

  // Watch for state transitions to trigger health checks
  useEffect(() => {
    const handleStateTransition = async () => {
      if (detectionSystem.detectionState === DetectionStates.READY) {
        const serviceStatus = detectionService.getDetailedStatus();
        
        if (!detectionSystem.healthCheckPerformed.current.postShutdown && 
            !serviceStatus.hasPerformedPostShutdownCheck) {
          console.log("🩺 Triggering post-shutdown health check...");
          await detectionSystem.performPostShutdownHealthCheck();
        } else if (!detectionSystem.healthCheckPerformed.current.initial && 
                   !serviceStatus.hasPerformedInitialHealthCheck) {
          console.log("🩺 Triggering initial health check...");
          await detectionSystem.performInitialHealthCheck();
        }
      }
    };

    handleStateTransition();
  }, [detectionSystem.detectionState]);

  // Enhanced camera detection
  const handleDetectCameras = useCallback(async () => {
    await cameraManagement.handleDetectCameras(lotManagement.showSnackbar);
    
    // Reset lot-related state if camera changed
    if (cameraManagement.selectedCameraId && 
        !cameraManagement.cameras.some(cam => cam.id.toString() === cameraManagement.selectedCameraId.toString())) {
      lotManagement.setCurrentLot(null);
      detectionSystem.setLastDetectionResult(null);
      detectionSystem.setIsStreamFrozen(false);
      // Clear lots cache since camera context changed
      invalidateLotsCache();
    }
  }, [cameraManagement, lotManagement, detectionSystem, invalidateLotsCache]);

  // Retry initialization
  const retryInitialization = createRetryInitialization(
    detectionSystem.initializationAttempted,
    detectionSystem.setInitializationError,
    detectionSystem.healthCheckPerformed
  );

  // Get display information
  const stateInfo = getStateInfo(detectionSystem.detectionState, detectionSystem.currentStreamingType);
  const modeInfo = getModeDisplayInfo(detectionSystem.currentStreamingType);
  const isBasicMode = detectionSystem.currentStreamingType === 'basic';
  const isDetectionRunning = detectionSystem.detectionState === DetectionStates.RUNNING;

  // Determine what to show in the right sidebar
  const showLotFormInSidebar = !isDetectionRunning && detectionSystem.detectionState === DetectionStates.READY;
  const showPerformancePanel = !showLotFormInSidebar || isPanelOpen;
  const showBasicControls = isBasicMode && isDetectionRunning && !isPanelOpen;

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* System Status Alerts */}
      {detectionSystem.detectionState === DetectionStates.INITIALIZING && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing adaptive detection system... Analyzing system capabilities.
        </Alert>
      )}

      {detectionSystem.detectionState === DetectionStates.SHUTTING_DOWN && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          System is shutting down... Please wait.
        </Alert>
      )}

      {/* Current Lot Information Alert */}
      {isBasicMode && lotManagement.currentLot && (
        <Alert 
          severity={lotManagement.currentLot.is_target_match ? "success" : "info"} 
          sx={{ mb: 2 }}
          icon={<Inventory />}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                ACTIVE LOT: {lotManagement.currentLot.lot_name}
              </Typography>
              <Typography variant="body2">
                Expected Piece ID: {lotManagement.currentLot.expected_piece_id} | 
                Number: {lotManagement.currentLot.expected_piece_number} | 
                Sessions: {lotManagement.currentLot.total_sessions || 0} | 
                Status: {lotManagement.currentLot.is_target_match ? 'Completed ✅' : 'Pending 🔄'}
              </Typography>
              {lotManagement.currentLot.completed_at && (
                <Typography variant="caption" color="textSecondary">
                  Completed: {new Date(lotManagement.currentLot.completed_at).toLocaleString()}
                </Typography>
              )}
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                size="small"
                icon={lotManagement.currentLot.is_target_match ? <CheckCircle /> : <Warning />}
                label={lotManagement.currentLot.is_target_match ? 'Complete' : 'Pending'}
                color={lotManagement.currentLot.is_target_match ? 'success' : 'warning'}
                variant="outlined"
              />
              <Tooltip title="Clear current lot">
                <IconButton 
                  size="small" 
                  onClick={() => {
                    lotManagement.setCurrentLot(null);
                    // Don't invalidate cache when just clearing current lot
                  }}
                >
                  <Close />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Alert>
      )}

      {/* System Mode Information */}
      {detectionSystem.detectionState === DetectionStates.READY && detectionSystem.systemProfile && (
        <Alert 
          severity="info" 
          sx={{ mb: 2 }}
          icon={modeInfo.icon === 'Smartphone' ? <Smartphone /> : <Computer />}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                {detectionSystem.currentStreamingType.toUpperCase()} MODE SELECTED
              </Typography>
              <Typography variant="body2">
                {modeInfo.description}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Performance Score: {detectionSystem.systemProfile.performance_score}/100 | 
                CPU: {detectionSystem.systemProfile.cpu_cores} cores | 
                RAM: {detectionSystem.systemProfile.available_memory_gb}GB | 
                GPU: {detectionSystem.systemProfile.gpu_available ? detectionSystem.systemProfile.gpu_name : 'None'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Refresh System Profile">
                <IconButton 
                  size="small" 
                  onClick={detectionHandlers.handleRefreshSystemProfile}
                  disabled={detectionSystem.isProfileRefreshing}
                >
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Run Performance Test">
                <IconButton 
                  size="small" 
                  onClick={detectionHandlers.handleRunPerformanceTest}
                  disabled={detectionSystem.detectionState !== DetectionStates.READY}
                >
                  <Speed />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Alert>
      )}

      {/* Error and Health Alerts */}
      {detectionSystem.initializationError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Adaptive system initialization failed: {detectionSystem.initializationError}
          <Box sx={{ mt: 1 }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={retryInitialization}
              disabled={detectionSystem.detectionState === DetectionStates.INITIALIZING || 
                       detectionSystem.detectionState === DetectionStates.SHUTTING_DOWN}
            >
              {detectionSystem.detectionState === DetectionStates.INITIALIZING ? 'Initializing...' : 'Retry Initialization'}
            </Button>
          </Box>
          <br />
          <small>If the issue persists, try refreshing the page or contact support.</small>
        </Alert>
      )}
      
      {!detectionSystem.systemHealth.overall && 
       detectionSystem.detectionState === DetectionStates.READY && 
       !detectionSystem.initializationError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          System health check indicates issues in {detectionSystem.currentStreamingType} mode. Detection may not work optimally.
          <br />
          <small>
            Streaming: {detectionSystem.systemHealth.streaming.status} | 
            Detection: {detectionSystem.systemHealth.detection.status} | 
            Last checked: {detectionSystem.getHealthCheckAge()}
          </small>
          <Box sx={{ mt: 1 }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={detectionHandlers.handleManualHealthCheck}
              disabled={detectionSystem.detectionState !== DetectionStates.READY}
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
              <DetectionControls
                targetLabel={targetLabel}
                onTargetLabelChange={detectionHandlers.handleTargetLabelChange}
                selectedCameraId={cameraManagement.selectedCameraId}
                onCameraChange={detectionHandlers.handleCameraChange}
                cameras={cameraManagement.cameras}
                onDetectCameras={handleDetectCameras}
                isDetecting={cameraManagement.isDetecting}
                isSystemReady={stateInfo.canOperate && detectionSystem.detectionState === DetectionStates.READY}
                systemHealth={detectionSystem.systemHealth}
                detectionOptions={detectionOptions}
                onDetectionOptionsChange={detectionHandlers.handleDetectionOptionsChange}
                detectionState={detectionSystem.detectionState}
              />
              
              <DetectionVideoFeed
                isDetectionActive={detectionSystem.detectionState === DetectionStates.RUNNING}
                onStartDetection={detectionHandlers.handleStartDetection}
                onStopDetection={detectionHandlers.handleStopDetection}
                cameraId={cameraManagement.cameraId}
                targetLabel={targetLabel}
                isSystemReady={stateInfo.canOperate}
                detectionOptions={detectionOptions}
                detectionState={detectionSystem.detectionState}
              />
            </Stack>
          </Box>
        </Grid>

        {/* Right Sidebar - Fixed Layout */}
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
            <Stack spacing={2} sx={{ height: '100%' }}>
              {/* Lot Form in Sidebar (when not detecting) */}
              {showLotFormInSidebar && (
                <Box sx={{ 
                  border: '1px solid', 
                  borderColor: 'divider', 
                  borderRadius: 1,
                  backgroundColor: 'background.paper',
                  overflow: 'hidden',
                  position: 'static',
                  zIndex: 'auto',
                  flex: '1 1 auto'
                }}>
                  <DetectionLotForm
                    isOpen={true}
                    isInSidebar={true}
                    onClose={() => {}}
                    onSubmit={handleLotFormSubmit}
                    onCreateLot={handleCreateLotOnly}  // Keep this - it's for "Create Only"
                    cameraId={cameraManagement.cameraId}
                    targetLabel={targetLabel}
                    detectionOptions={detectionOptions}
                    isSubmitting={lotManagement.lotOperationInProgress}
                    existingLots={lotManagement.existingLots}
                    onRefreshLots={lotManagement.fetchExistingLots}
                  />
                </Box>
              )}

              {/* Basic Mode Detection Controls (when detecting) */}
              {showBasicControls && (
                <Box sx={{ 
                  position: 'static', // Use static positioning
                  zIndex: 'auto', // Remove high z-index
                  maxWidth: '100%', // Use full width of sidebar
                  flex: '1 1 auto'
                }}>
                  <BasicModeControls
                    isStreamFrozen={detectionSystem.isStreamFrozen}
                    onDemandDetecting={detectionSystem.onDemandDetecting}
                    detectionInProgress={detectionSystem.detectionInProgress}
                    lastDetectionResult={detectionSystem.lastDetectionResult}
                    targetLabel={targetLabel}
                    onOnDemandDetection={detectionHandlers.handleOnDemandDetection}
                    onFreezeStream={detectionHandlers.handleFreezeStream}
                    onUnfreezeStream={detectionHandlers.handleUnfreezeStream}
                    currentLot={lotManagement.currentLot}
                    lotOperationInProgress={lotManagement.lotOperationInProgress}
                    onOpenLotForm={() => {}}
                  />
                </Box>
              )}
              
              {/* System Performance Panel */}
              {showPerformancePanel && (
                <Box sx={{
                  position: 'static',
                  zIndex: 'auto',
                  flex: showLotFormInSidebar ? '0 0 auto' : '1 1 auto' // Take remaining space when form is not shown
                }}>
                  <SystemPerformancePanel
                    detectionState={detectionSystem.detectionState}
                    systemHealth={detectionSystem.systemHealth}
                    globalStats={detectionSystem.globalStats}
                    detectionOptions={detectionOptions}
                    healthCheckPerformed={detectionSystem.healthCheckPerformed}
                    autoModeEnabled={detectionSystem.autoModeEnabled}
                    isBasicMode={isBasicMode}
                    getHealthCheckAge={detectionSystem.getHealthCheckAge}
                    handleManualHealthCheck={detectionHandlers.handleManualHealthCheck}
                    handleSwitchToBasicMode={detectionHandlers.handleSwitchToBasicMode}
                    handleSwitchToOptimizedMode={detectionHandlers.handleSwitchToOptimizedMode}
                    handleEnableAutoMode={detectionHandlers.handleEnableAutoMode}
                    DetectionStates={DetectionStates}
                    isPanelOpen={isPanelOpen}
                    onPanelToggle={() => setIsPanelOpen(!isPanelOpen)}
                    isDetectionRunning={isDetectionRunning}
                  />
                </Box>
              )}
            </Stack>
          </Box>
        </Grid>
      </Grid>

      {/* Snackbar for notifications */}
      <Snackbar
        open={lotManagement.snackbar.open}
        autoHideDuration={6000}
        onClose={() => lotManagement.setSnackbar(prev => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => lotManagement.setSnackbar(prev => ({ ...prev, open: false }))}
          severity={lotManagement.snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {lotManagement.snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}