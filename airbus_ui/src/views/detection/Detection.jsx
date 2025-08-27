// AppDetection.jsx - UPDATED: Proper lot-based initialization
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
import DetectionControls from "./components/DetectionControls";
import DetectionVideoFeed from "./components/DetectionVideoFeed";
import SystemPerformancePanel from "./components/SystemPerformancePanel";
import LotWorkflowPanel from "./components/LotWorkflowPanel";
import InfoPanel from "./components/InfoPanel";
import DetectionStatsPanel from "./components/DetectionStatsPanel";

// Services
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
} from "./handlers/AppDetectionHandlers";

// Utils
import { 
  getStateInfo,
  getModeDisplayInfo,
  createCleanupFunction,
  DetectionStates
} from "./utils/AppDetectionUtils";

export default function AppDetection() {
  const [searchParams, setSearchParams] = useSearchParams();
  
  // UI State
  const [targetLabel, setTargetLabel] = useState("");
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isStatsPanelOpen, setIsStatsPanelOpen] = useState(false);
  
  // Detection Options
  const [detectionOptions, setDetectionOptions] = useState({
    detectionFps: 5.0,
    streamQuality: 85,
    priority: 1,
    enableAdaptiveQuality: true,
    enableFrameSkipping: true,
    quality: 85
  });

  // Enhanced lot workflow state with proper initialization tracking
  const [selectedLotId, setSelectedLotId] = useState(null);
  const [lotWorkflowActive, setLotWorkflowActive] = useState(false);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [isLotLoading, setIsLotLoading] = useState(false);
  const [lotLoadInitialized, setLotLoadInitialized] = useState(false);
  
  // NEW: Lot-based initialization state
  const [isInitializingForLot, setIsInitializingForLot] = useState(false);
  const [initializationError, setInitializationError] = useState(null);
  const [lotInitialized, setLotInitialized] = useState(false);

  // SIMPLIFIED: Only one trigger for successful detections - timestamp based
  const [lastSuccessfulDetection, setLastSuccessfulDetection] = useState(null);

  // Piece labels cache
  const [pieceLabels, setPieceLabels] = useState(new Map());
  const [loadingLabels, setLoadingLabels] = useState(new Set());

  // Caching system
  const lotsCache = useRef({
    data: new Map(),
    lastFetch: 0,
    isValid: false,
    fetchPromise: null
  });
  
  const LOTS_CACHE_DURATION = 30000; // 30 seconds cache

  // Custom hooks for state management
  const lotManagement = useLotManagement(selectedLotId, lotWorkflowActive);
  const detectionSystem = useDetectionSystem();
  const cameraManagement = useCameraManagement();

  // Get StreamManager instance from detection service
  const streamManager = detectionService.streamManager;

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

  // Function to get piece label with loading state
  const getPieceLabel = useCallback((pieceId) => {
    if (pieceLabels.has(pieceId)) {
      return pieceLabels.get(pieceId);
    }
    
    if (loadingLabels.has(pieceId)) {
      return 'Loading...';
    }
    
    // Trigger fetch
    fetchPieceLabel(pieceId);
    return `Piece ${pieceId}`;
  }, [pieceLabels, loadingLabels, fetchPieceLabel]);

  // UPDATED: Trigger stats refresh only on successful detection
  const triggerStatsRefreshOnSuccess = useCallback(() => {
    console.log('Detection successful - triggering stats refresh');
    setLastSuccessfulDetection(Date.now()); // Use timestamp to ensure uniqueness
  }, []);

  // NEW: Lot-based initialization function
  const initializeDetectionForLot = useCallback(async (lotId, pieceLabel) => {
    console.log(`Initializing detection system for lot ${lotId} with piece: ${pieceLabel}`);
    
    setIsInitializingForLot(true);
    setInitializationError(null);
    
    try {
      // Use the new lot-based initialization from DetectionService
      const result = await detectionService.initializeForLot(lotId, pieceLabel);
      
      if (result.success) {
        setLotInitialized(true);
        console.log(`Detection system ready for ${pieceLabel}`);
        
        lotManagement.showSnackbar(
          `Detection system initialized for ${pieceLabel}`, 
          'success'
        );
        
        return { success: true };
      } else {
        throw new Error(result.message || 'Failed to initialize for lot');
      }
      
    } catch (error) {
      console.error(`Failed to initialize for lot ${lotId}:`, error);
      setInitializationError(error.message);
      setLotInitialized(false);
      
      lotManagement.showSnackbar(
        `Failed to initialize for ${pieceLabel}: ${error.message}`, 
        'error'
      );
      
      return { success: false, error: error.message };
    } finally {
      setIsInitializingForLot(false);
    }
  }, [lotManagement]);

  // Enhanced lot loading function with proper initialization
  const loadSelectedLot = useCallback(async (forceReload = false) => {
    if (!selectedLotId) {
      console.log('No lot ID to load');
      setLotLoadInitialized(true);
      return { success: false, message: 'No lot ID provided' };
    }

    // Don't reload if we already have the correct lot loaded unless forced
    if (!forceReload && 
        lotManagement.currentLot && 
        lotManagement.currentLot.lot_id === selectedLotId &&
        lotLoadInitialized) {
      console.log('Lot already loaded correctly:', lotManagement.currentLot.lot_name);
      return { success: true, lot: lotManagement.currentLot };
    }

    setIsLotLoading(true);
    
    try {
      console.log('Loading selected lot:', selectedLotId);
      
      // Ensure StreamManager is available
      if (!streamManager) {
        throw new Error('StreamManager not available');
      }
      
      // Get lot details from StreamManager
      const lotResult = await streamManager.getDetectionLot(selectedLotId);
      
      if (lotResult.success && (lotResult.lotData || lotResult.lot)) {
        const lot = lotResult.lotData || lotResult.lot;
        
        // Fetch the piece label from API and add it to the lot object
        let pieceLabel = lot.expected_piece_label;
        if (!pieceLabel && lot.expected_piece_id) {
          console.log('Fetching piece label for ID:', lot.expected_piece_id);
          pieceLabel = await fetchPieceLabel(lot.expected_piece_id);
          lot.expected_piece_label = pieceLabel;
        }
        
        // Set the current lot with the enhanced piece label
        lotManagement.setCurrentLot(lot);
        
        // Set target label to the actual piece label
        const targetLabelToUse = pieceLabel || `piece_${lot.expected_piece_id}`;
        setTargetLabel(targetLabelToUse);
        
        // CRITICAL: Initialize detection system for this specific lot and piece
        console.log(`Initializing detection system for lot ${selectedLotId} with piece: ${targetLabelToUse}`);
        const initResult = await initializeDetectionForLot(selectedLotId, targetLabelToUse);
        
        if (initResult.success) {
          // Activate lot workflow since we have a selected lot and initialized system
          setLotWorkflowActive(true);
          
          // Load detection history for this lot
          try {
            const historyResult = await streamManager.getLotDetectionSessions(selectedLotId);
            if (historyResult.success) {
              setDetectionHistory(historyResult.sessions || []);
              console.log(`Loaded ${historyResult.sessions?.length || 0} detection sessions`);
            }
          } catch (historyError) {
            console.warn('Failed to load detection history:', historyError);
            setDetectionHistory([]);
          }
          
          // Mark as initialized
          setLotLoadInitialized(true);
          
          console.log('Lot loaded and system initialized successfully:', {
            lotId: lot.lot_id,
            lotName: lot.lot_name,
            expectedPiece: targetLabelToUse,
            isComplete: lot.is_target_match,
            workflowActive: true,
            systemInitialized: true
          });
          
          return { success: true, lot };
        } else {
          throw new Error(`Failed to initialize detection system: ${initResult.error}`);
        }
      } else {
        throw new Error(`Lot not found or API error: ${lotResult.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error loading selected lot:', error);
      
      let errorMessage = `Failed to load lot: ${error.message}`;
      lotManagement.showSnackbar(errorMessage, 'error');
      
      // Mark as initialized even on error to prevent infinite loops
      setLotLoadInitialized(true);
      setLotInitialized(false);
      
      return { success: false, error: errorMessage };
    } finally {
      setIsLotLoading(false);
    }
  }, [selectedLotId, lotManagement, streamManager, setTargetLabel, fetchPieceLabel, lotLoadInitialized, initializeDetectionForLot]);

  // Enhanced URL parameter handling
  useEffect(() => {
    const lotIdFromUrl = searchParams.get('lotId');
    const modeFromUrl = searchParams.get('mode');
    
    console.log('URL params check:', { 
      lotIdFromUrl, 
      modeFromUrl, 
      currentSelectedLotId: selectedLotId,
      streamManagerAvailable: !!streamManager 
    });
    
    if (lotIdFromUrl && parseInt(lotIdFromUrl) !== selectedLotId) {
      console.log('New lot selected from URL:', lotIdFromUrl);
      
      // Reset lot-specific states
      const newLotId = parseInt(lotIdFromUrl);
      setSelectedLotId(newLotId);
      setLotLoadInitialized(false);
      setLotInitialized(false);
      setIsInitializingForLot(false);
      setInitializationError(null);
    }
  }, [searchParams, selectedLotId, streamManager]);

  // UPDATED: Load lot only when needed, without waiting for system state
  useEffect(() => {
    if (selectedLotId && 
        streamManager && 
        !lotLoadInitialized &&
        !isLotLoading) {
      
      console.log('Conditions met, loading lot details:', {
        selectedLotId,
        lotLoadInitialized,
        isLotLoading
      });
      
      loadSelectedLot();
    }
  }, [selectedLotId, streamManager, lotLoadInitialized, isLotLoading, loadSelectedLot]);

  // Enhanced lots fetching with improved caching
  const fetchExistingLotsEfficient = useCallback(async (forceRefresh = false) => {
    const now = Date.now();
    
    if (!forceRefresh && 
        lotsCache.current.isValid && 
        lotsCache.current.data.size > 0 && 
        (now - lotsCache.current.lastFetch) < LOTS_CACHE_DURATION) {
      console.log('Using cached lots data');
      const cachedData = Array.from(lotsCache.current.data.values());
      lotManagement.setExistingLots(cachedData);
      return cachedData;
    }

    if (lotsCache.current.fetchPromise) {
      console.log('Fetch already in progress, waiting...');
      try {
        return await lotsCache.current.fetchPromise;
      } catch (error) {
        console.warn('Previous fetch failed, retrying');
        lotsCache.current.fetchPromise = null;
      }
    }

    // Don't fetch if StreamManager is not available yet
    if (!streamManager) {
      console.log('StreamManager not available, skipping lots fetch');
      return [];
    }

    lotsCache.current.fetchPromise = (async () => {
      try {
        console.log('Fetching lots via StreamManager...');

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
          
          console.log(`Fetched ${result.lots.length} lots`);
          return result.lots;
        } else {
          throw new Error(result.message || 'Failed to fetch lots');
        }

      } catch (error) {
        console.error('Error fetching lots:', error);
        lotsCache.current.isValid = false;
        
        lotManagement.showSnackbar(`Failed to fetch lots: ${error.message}`, 'error');
        
        return lotManagement.existingLots;
      } finally {
        lotsCache.current.fetchPromise = null;
      }
    })();

    return lotsCache.current.fetchPromise;
  }, [streamManager, lotManagement]);

  // Enhanced detection handlers with proper lot context
  const detectionHandlers = useDetectionHandlers({
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
    selectedLotId,
    lotWorkflowActive,
    detectionHistory,
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
    setDetectionHistory,
    showSnackbar: lotManagement.showSnackbar,
    fetchExistingLots: fetchExistingLotsEfficient,
    performSingleHealthCheck: detectionSystem.performSingleHealthCheck,
    performPostShutdownHealthCheck: detectionSystem.performPostShutdownHealthCheck,
    loadSelectedLot,
    
    // Enhanced functions
    streamManager,
  });

  // Enhanced lot workflow handlers
  const handleStartLotWorkflow = useCallback(async () => {
    console.log('Starting lot workflow...', {
      selectedLotId,
      hasCurrentLot: !!lotManagement.currentLot,
      cameraId: cameraManagement.cameraId,
      targetLabel,
      lotInitialized
    });

    if (!lotInitialized) {
      lotManagement.showSnackbar('Detection system not initialized for this lot. Please wait for initialization to complete.', 'error');
      return;
    }

    if (!selectedLotId || !lotManagement.currentLot) {
      lotManagement.showSnackbar('Please select a lot first', 'error');
      return;
    }

    if (!cameraManagement.cameraId) {
      lotManagement.showSnackbar('Please select a camera first', 'error');
      return;
    }

    setLotWorkflowActive(true);
    console.log('Starting detection with lot context and initialized system');
    await detectionHandlers.handleStartDetection();
  }, [selectedLotId, lotManagement.currentLot, cameraManagement.cameraId, detectionHandlers, lotInitialized]);

  const handleStopLotWorkflow = useCallback(async () => {
    console.log('Stopping lot workflow');
    
    setLotWorkflowActive(false);
    setSelectedLotId(null);
    setDetectionHistory([]);
    setLotLoadInitialized(false);
    setLotInitialized(false);
    setIsInitializingForLot(false);
    setInitializationError(null);
    
    // Shutdown lot-specific initialization
    try {
      await detectionService.shutdownLotSpecificInitialization();
    } catch (error) {
      console.warn('Error during lot-specific shutdown:', error);
    }
    
    searchParams.delete('lotId');
    searchParams.delete('mode');
    setSearchParams(searchParams, { replace: true });
    
    await detectionHandlers.handleStopDetection();
  }, [detectionHandlers, searchParams, setSearchParams]);

  // UPDATED: Enhanced lot detection with proper success trigger
  const handleLotDetection = useCallback(async () => {
    console.log('Lot detection requested...', {
      selectedLotId,
      hasCurrentLot: !!lotManagement.currentLot,
      cameraId: cameraManagement.cameraId,
      targetLabel,
      lotInitialized
    });

    if (!lotInitialized) {
      lotManagement.showSnackbar('Detection system not initialized for this lot. Please wait.', 'error');
      return;
    }

    if (!selectedLotId || !cameraManagement.cameraId || !targetLabel) {
      lotManagement.showSnackbar('Lot, camera, and target label are required', 'error');
      return;
    }

    try {
      detectionSystem.setDetectionInProgress(true);
      
      console.log(`Performing detection for lot ${selectedLotId}`);
      
      const result = await detectionHandlers.handleLotWorkflowDetection();

      if (result) {
        // CRITICAL: Only trigger stats refresh on successful detection
        console.log('Detection successful - triggering stats refresh and opening panel');
        triggerStatsRefreshOnSuccess();
        
        // Reload lot data to get updated information
        await loadSelectedLot(true);
        
        if (result.lotCompleted) {
          lotManagement.showSnackbar('Lot completed successfully! All requirements met.', 'success');
        } else {
          lotManagement.showSnackbar(
            result.detectionResult?.detected_target 
              ? 'Target detected but lot needs verification. Continue detecting until correct.'
              : 'Target not found. Please adjust the piece and try again.',
            'info'
          );
        }
      } else {
        // Don't trigger refresh on failed detection
        console.log('Detection failed - no stats refresh needed');
        lotManagement.showSnackbar('Detection failed. Please try again.', 'error');
      }
    } catch (error) {
      console.error('Error in lot detection:', error);
      lotManagement.showSnackbar(`Detection failed: ${error.message}`, 'error');
      // Don't trigger refresh on error
    } finally {
      detectionSystem.setDetectionInProgress(false);
    }
  }, [selectedLotId, cameraManagement.cameraId, targetLabel, lotManagement, detectionSystem, loadSelectedLot, detectionHandlers, triggerStatsRefreshOnSuccess, lotInitialized]);

  // Load initial lots once (only when StreamManager is available)
  useEffect(() => {
    let mounted = true;
    
    const loadInitialLots = async () => {
      if (mounted && !lotsCache.current.isValid && streamManager) {
        console.log('Loading initial lots...');
        await fetchExistingLotsEfficient(false);
      }
    };
    
    loadInitialLots();
    
    return () => {
      mounted = false;
    };
  }, [streamManager, fetchExistingLotsEfficient]);

  // REMOVED: Old system initialization effect - no longer needed
  // The system only initializes when a lot is selected

  // Enhanced cleanup
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
      lotsCache.current.data.clear();
      lotsCache.current.isValid = false;
      if (lotsCache.current.fetchPromise) {
        lotsCache.current.fetchPromise = null;
      }
    };
  }, [detectionSystem.detectionState, detectionSystem.stopMonitoring, detectionSystem.cleanupRef]);

  // Enhanced camera detection
  const handleDetectCameras = useCallback(async () => {
    await cameraManagement.handleDetectCameras(lotManagement.showSnackbar);
    
    if (cameraManagement.selectedCameraId && 
        !cameraManagement.cameras.some(cam => cam.id.toString() === cameraManagement.selectedCameraId.toString())) {
      if (!lotWorkflowActive) {
        lotManagement.setCurrentLot(null);
      }
      detectionSystem.setLastDetectionResult(null);
      detectionSystem.setIsStreamFrozen(false);
    }
  }, [cameraManagement, lotManagement, detectionSystem, lotWorkflowActive]);
  
  // NEW: Handle retry initialization for lot
  const handleRetryLotInitialization = useCallback(async () => {
    if (selectedLotId && targetLabel) {
      setInitializationError(null);
      await initializeDetectionForLot(selectedLotId, targetLabel);
    }
  }, [selectedLotId, targetLabel, initializeDetectionForLot]);

  const stateInfo = getStateInfo(detectionSystem.detectionState, detectionSystem.currentStreamingType);
  const modeInfo = getModeDisplayInfo(detectionSystem.currentStreamingType);
  const isBasicMode = detectionSystem.currentStreamingType === 'basic';
  const isDetectionRunning = detectionSystem.detectionState === DetectionStates.RUNNING;
  
  // Only show LotWorkflowPanel in basic mode and when lot is initialized
  const showLotWorkflowPanel = lotWorkflowActive && selectedLotId && isBasicMode && lotInitialized;
  const showLotFormInSidebar = !isDetectionRunning && !showLotWorkflowPanel;
  const showPerformancePanel = true;
  const showStatsPanel = true;

  // Debug logging
  useEffect(() => {
    console.log('Component state debug:', {
      selectedLotId,
      lotWorkflowActive,
      hasCurrentLot: !!lotManagement.currentLot,
      currentLotId: lotManagement.currentLot?.lot_id,
      currentLotName: lotManagement.currentLot?.lot_name,
      detectionState: detectionSystem.detectionState,
      currentStreamingType: detectionSystem.currentStreamingType,
      targetLabel,
      isLotLoading,
      lotLoadInitialized,
      streamManagerAvailable: !!streamManager,
      isBasicMode,
      showLotWorkflowPanel,
      isStreamFrozen: detectionSystem.isStreamFrozen,
      // NEW: Lot initialization state
      isInitializingForLot,
      lotInitialized,
      initializationError
    });
  }, [selectedLotId, lotWorkflowActive, lotManagement.currentLot, detectionSystem.detectionState, detectionSystem.currentStreamingType, targetLabel, isLotLoading, lotLoadInitialized, streamManager, isBasicMode, showLotWorkflowPanel, detectionSystem.isStreamFrozen, isInitializingForLot, lotInitialized, initializationError]);

return (
  <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
    {/* Clean Info Panel */}
    <InfoPanel
      showLotWorkflowPanel={showLotWorkflowPanel && lotManagement.currentLot}
      currentLot={lotManagement.currentLot}
      detectionHistory={detectionHistory}
      getPieceLabel={getPieceLabel}
      onStopLotWorkflow={handleStopLotWorkflow}
      isBasicMode={isBasicMode && lotManagement.currentLot && !showLotWorkflowPanel}
      detectionState={detectionSystem.detectionState}
      systemProfile={detectionSystem.systemProfile}
      currentStreamingType={detectionSystem.currentStreamingType}
      isProfileRefreshing={detectionSystem.isProfileRefreshing}
      onRefreshSystemProfile={detectionHandlers.handleRefreshSystemProfile}
      onRunPerformanceTest={detectionHandlers.handleRunPerformanceTest}
      DetectionStates={DetectionStates}
    />

    {/* Detection Statistics Panel */}
    {showStatsPanel && (
      <Box sx={{ mb: 2 }}>
        <DetectionStatsPanel
          isOpen={isStatsPanelOpen}
          onToggle={setIsStatsPanelOpen}
          isDetectionActive={isDetectionRunning}
          onStartDetection={lotWorkflowActive ? handleStartLotWorkflow : detectionHandlers.handleStartDetection}
          currentLotId={selectedLotId}
          detectionCompleted={lastSuccessfulDetection}
          isStreamFrozen={detectionSystem.isStreamFrozen}
        />
      </Box>
    )}

    {/* NEW: Lot initialization status alerts */}
    {isLotLoading && (
      <Alert 
        severity="info" 
        sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
        icon={<CircularProgress size={20} />}
      >
        Loading lot details for lot {selectedLotId}...
      </Alert>
    )}

    {isInitializingForLot && (
      <Alert 
        severity="info" 
        sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
        icon={<CircularProgress size={20} />}
      >
        Initializing detection system for piece: {targetLabel}... This may take a moment as the model is being loaded.
      </Alert>
    )}

    {initializationError && (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to initialize detection system for {targetLabel}: {initializationError}
        <Box sx={{ mt: 1 }}>
          <Button 
            variant="outlined" 
            size="small" 
            onClick={handleRetryLotInitialization}
            disabled={isInitializingForLot}
          >
            {isInitializingForLot ? 'Initializing...' : 'Retry Initialization'}
          </Button>
        </Box>
      </Alert>
    )}

    {lotInitialized && lotManagement.currentLot && (
      <Alert severity="success" sx={{ mb: 2 }}>
        Detection system ready for {targetLabel}! You can now start detection.
      </Alert>
    )}

    {/* Keep only the System Status Alerts for general system issues */}
    {detectionSystem.detectionState === DetectionStates.SHUTTING_DOWN && (
      <Alert 
        severity="warning" 
        sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
        icon={<CircularProgress size={20} />}
      >
        System is shutting down... Please wait.
      </Alert>
    )}

    {/* Keep health alerts - these are still important */}
    {!detectionSystem.systemHealth.overall && 
     detectionSystem.detectionState === DetectionStates.READY && 
     !initializationError && (
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
                isSystemReady={lotInitialized && lotManagement.currentLot && detectionSystem.detectionState === DetectionStates.READY}
                systemHealth={detectionSystem.systemHealth}
                detectionOptions={detectionOptions}
                onDetectionOptionsChange={detectionHandlers.handleDetectionOptionsChange}
                detectionState={detectionSystem.detectionState}
                selectedLotId={selectedLotId}
                lotWorkflowActive={lotWorkflowActive}
                currentLot={lotManagement.currentLot}
              />
              
              <DetectionVideoFeed
                isDetectionActive={detectionSystem.detectionState === DetectionStates.RUNNING}
                onStartDetection={lotWorkflowActive ? handleStartLotWorkflow : detectionHandlers.handleStartDetection}
                onStopDetection={lotWorkflowActive ? handleStopLotWorkflow : detectionHandlers.handleStopDetection}
                cameraId={cameraManagement.cameraId}
                targetLabel={targetLabel}
                isSystemReady={lotInitialized && lotManagement.currentLot}
                detectionOptions={detectionOptions}
                detectionState={detectionSystem.detectionState}
                lotWorkflowActive={lotWorkflowActive}
                selectedLotId={selectedLotId}
                currentLot={lotManagement.currentLot}
                navigateOnStop={lotWorkflowActive ? handleStopLotWorkflow : detectionHandlers.handleStopDetection}
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
              
              {/* Lot Workflow Panel (highest priority) */}
              {showLotWorkflowPanel && (
                <Box sx={{ 
                  position: 'static',
                  zIndex: 'auto',
                  maxWidth: '100%',
                  flex: '1 1 auto'
                }}>
                  <LotWorkflowPanel
                    currentLot={lotManagement.currentLot}
                    selectedLotId={selectedLotId}
                    detectionHistory={detectionHistory}
                    detectionInProgress={detectionSystem.detectionInProgress}
                    onDemandDetecting={detectionSystem.onDemandDetecting}
                    lastDetectionResult={detectionSystem.lastDetectionResult}
                    isStreamFrozen={detectionSystem.isStreamFrozen}
                    targetLabel={targetLabel}
                    onDetectLot={handleLotDetection}
                    onFreezeStream={detectionHandlers.handleFreezeStream}
                    onUnfreezeStream={detectionHandlers.handleUnfreezeStream}
                    onStopWorkflow={handleStopLotWorkflow}
                    onReloadHistory={() => loadSelectedLot(true)}
                    streamManager={streamManager}
                  />
                </Box>
              )}
              
              {/* System Performance Panel */}
              {showPerformancePanel && (
                <Box sx={{
                  position: 'static',
                  zIndex: 'auto',
                  flex: showLotFormInSidebar ? '0 0 auto' : '1 1 auto'
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