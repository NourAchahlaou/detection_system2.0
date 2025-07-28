// DetectionVideoFeed.jsx - FIXED: Single initialization only
import React, { useState, useEffect, useRef, useCallback } from "react";
import { Box, Alert, CircularProgress, Typography, Chip } from "@mui/material";
import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";
import { detectionService } from "../detectionService";

// GLOBAL state to prevent multiple initializations across all components
let globalInitializationState = {
  isInitialized: false,
  isInitializing: false,
  initializationPromise: null,
  error: null,
  modelLoaded: false
};

const DetectionVideoFeed = ({
  isDetectionActive,
  onStartDetection,
  onStopDetection,
  cameraId,
  targetLabel,
  isModelLoaded,
  onModelLoadedChange = () => {},
  detectionOptions = {}
}) => {
  const [videoUrl, setVideoUrl] = useState("");
  const [showControls, setShowControls] = useState(false);
  const [detectionStats, setDetectionStats] = useState({
    objectDetected: false,
    detectionCount: 0,
    nonTargetCount: 0,
    lastDetectionTime: null,
    avgProcessingTime: 0,
    streamQuality: 85,
    detectionFps: 5.0,
    queueDepth: 0,
    isStreamActive: false
  });
  const [streamStatus, setStreamStatus] = useState({
    isLoading: false,
    error: null,
    isConnected: false
  });
  
  // Use global state for initialization
  const [initializationStatus, setInitializationStatus] = useState({
    isInitializing: globalInitializationState.isInitializing,
    initializationError: globalInitializationState.error,
    initializationAttempts: 0
  });
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);

  // SINGLE initialization function that runs only once globally
  const initializeDetectionSystemOnce = useCallback(async () => {
    // If already initialized or initializing, don't do anything
    if (globalInitializationState.isInitialized || globalInitializationState.isInitializing) {
      console.log('🔄 Detection system already initialized or initializing, skipping...');
      
      // If initialization is in progress, wait for it
      if (globalInitializationState.initializationPromise) {
        try {
          await globalInitializationState.initializationPromise;
        } catch (error) {
          console.error('Waiting for existing initialization failed:', error);
        }
      }
      
      // Update local state to match global state
      if (mountedRef.current) {
        setInitializationStatus({
          isInitializing: globalInitializationState.isInitializing,
          initializationError: globalInitializationState.error,
          initializationAttempts: 0
        });
        
        onModelLoadedChange(globalInitializationState.modelLoaded);
      }
      return;
    }

    console.log('🚀 Starting SINGLE detection system initialization...');
    
    // Set global state
    globalInitializationState.isInitializing = true;
    globalInitializationState.error = null;
    
    // Update local state
    if (mountedRef.current) {
      setInitializationStatus(prev => ({ 
        ...prev, 
        isInitializing: true, 
        initializationError: null 
      }));
    }

    // Create initialization promise
    globalInitializationState.initializationPromise = (async () => {
      try {
        console.log("🔧 Initializing detection processor...");
        
        // Initialize processor (this should be idempotent)
        const initResult = await detectionService.ensureInitialized();
        console.log("✅ Processor initialization result:", initResult);

        // Load model (this should also be idempotent)
        console.log("🤖 Loading detection model...");
        const modelResult = await detectionService.loadModel();
        console.log("✅ Model loading result:", modelResult);
        
        // Update global state
        globalInitializationState.isInitialized = true;
        globalInitializationState.modelLoaded = modelResult.success;
        globalInitializationState.isInitializing = false;
        globalInitializationState.error = null;
        
        // Update local state if component is still mounted
        if (mountedRef.current) {
          setInitializationStatus({
            isInitializing: false,
            initializationError: null,
            initializationAttempts: 0
          });
          
          // Notify parent component
          onModelLoadedChange(modelResult.success);
        }
        
        console.log("✅ SINGLE detection system initialization completed successfully");
        return { success: true };

      } catch (error) {
        console.error("❌ SINGLE detection system initialization failed:", error);
        
        // Update global state
        globalInitializationState.isInitializing = false;
        globalInitializationState.error = error.message;
        globalInitializationState.isInitialized = false;
        globalInitializationState.modelLoaded = false;
        
        // Update local state if component is still mounted
        if (mountedRef.current) {
          setInitializationStatus(prev => ({ 
            ...prev, 
            isInitializing: false,
            initializationError: error.message,
            initializationAttempts: prev.initializationAttempts + 1
          }));
          
          // Notify parent component about failure
          onModelLoadedChange(false);
        }
        
        throw error;
      } finally {
        // Clear the promise reference when done
        globalInitializationState.initializationPromise = null;
      }
    })();

    // Wait for initialization to complete
    try {
      await globalInitializationState.initializationPromise;
    } catch (error) {
      // Error already handled above
    }
  }, [onModelLoadedChange]);

  // Initialize ONLY ONCE when component mounts
  useEffect(() => {
    mountedRef.current = true;
    
    // Initialize detection system (will only happen once globally)
    initializeDetectionSystemOnce();
    
    return () => {
      mountedRef.current = false;
    };
  }, []); // Empty dependency array - run ONLY once on mount

  // Reset global state function (for manual retry)
  const resetAndRetryInitialization = useCallback(async () => {
    console.log("🔄 Manually resetting and retrying detection system initialization...");
    
    // Reset global state completely
    globalInitializationState = {
      isInitialized: false,
      isInitializing: false,
      initializationPromise: null,
      error: null,
      modelLoaded: false
    };
    
    // Reset local state
    setInitializationStatus({
      isInitializing: false,
      initializationError: null,
      initializationAttempts: 0
    });
    
    // Try initialization again
    await initializeDetectionSystemOnce();
  }, [initializeDetectionSystemOnce]);

  // Stats listener callback
  const handleStatsUpdate = useCallback((newStats) => {
    if (!mountedRef.current) return;
    
    setDetectionStats(prevStats => ({
      ...prevStats,
      ...newStats,
      detectionCount: newStats.objectDetected && !prevStats.objectDetected 
        ? prevStats.detectionCount + 1 
        : (newStats.detectionCount || prevStats.detectionCount)
    }));
  }, []);

  // Enhanced detection start with proper initialization check
  const handleStartDetection = async () => {
    if (!cameraId || cameraId === '') {
      alert("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      alert("Please enter a target label first.");
      return;
    }

    // Check global initialization state
    if (globalInitializationState.isInitializing) {
      alert("Detection system is still initializing. Please wait...");
      return;
    }

    // If there's an initialization error, offer to retry
    if (globalInitializationState.error) {
      const shouldRetry = window.confirm(
        `Detection system initialization failed: ${globalInitializationState.error}\n\nWould you like to retry initialization?`
      );
      if (shouldRetry) {
        await resetAndRetryInitialization();
        return;
      } else {
        return;
      }
    }
    
    if (!globalInitializationState.modelLoaded) {
      alert("Detection model is not loaded. Please wait for initialization to complete or try refreshing the page.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });

    try {
      const streamUrl = await detectionService.startOptimizedDetectionFeed(
        parseInt(cameraId), 
        targetLabel, 
        {
          detectionFps: detectionOptions.detectionFps || 5.0,
          streamQuality: detectionOptions.streamQuality || 85,
          priority: detectionOptions.priority || 1
        }
      );

      if (!mountedRef.current) return;

      setVideoUrl(streamUrl);
      detectionService.addStatsListener(parseInt(cameraId), handleStatsUpdate);
      
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: detectionOptions.streamQuality || 85,
        detectionFps: detectionOptions.detectionFps || 5.0,
        queueDepth: 0,
        isStreamActive: true
      });

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      onStartDetection();

      console.log(`✅ Started optimized detection for camera ${cameraId} with target: ${targetLabel}`);

    } catch (error) {
      console.error("❌ Error starting optimized detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus({ 
        isLoading: false, 
        error: error.message || "Failed to start detection", 
        isConnected: false 
      });
      alert(`Failed to start detection: ${error.message}`);
    }
  };

  // Handle stopping detection
  const handleStopDetection = async () => {
    if (!mountedRef.current) return;
    
    setStreamStatus(prev => ({ ...prev, isLoading: true }));

    try {
      detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      await detectionService.stopOptimizedDetectionFeed(parseInt(cameraId), false); // Don't shutdown system
      
      if (!mountedRef.current) return;
      
      setVideoUrl("");
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: 85,
        detectionFps: 5.0,
        queueDepth: 0,
        isStreamActive: false
      });
      
      setStreamStatus({ isLoading: false, error: null, isConnected: false });
      onStopDetection();

      console.log(`✅ Stopped optimized detection for camera ${cameraId}`);

    } catch (error) {
      console.error("❌ Error stopping optimized detection:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop detection" }));
      alert(`Failed to stop detection: ${error.message}`);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraId) {
        detectionService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
    };
  }, [cameraId, handleStatsUpdate]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* Initialization Status Alerts */}
      {initializationStatus.isInitializing && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing detection system... This happens only once when you open the page.
        </Alert>
      )}

      {initializationStatus.initializationError && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, width: '100%' }}
          action={
            <button 
              onClick={resetAndRetryInitialization}
              style={{ 
                background: 'none', 
                border: '1px solid currentColor', 
                color: 'inherit',
                padding: '4px 8px',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          }
        >
          Detection system initialization failed: {initializationStatus.initializationError}
        </Alert>
      )}

      {/* Stream Status Alerts */}
      {streamStatus.error && (
        <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
          {streamStatus.error}
        </Alert>
      )}

      {streamStatus.isLoading && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {isDetectionActive ? "Connecting to optimized detection stream..." : "Stopping stream..."}
        </Alert>
      )}

      <VideoCard
        cameraActive={isDetectionActive}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isDetectionActive ? (
          <LiveDetectionView
            videoUrl={videoUrl}
            videoRef={videoRef}
            showControls={showControls}
            onStopDetection={handleStopDetection}
            detectionStats={detectionStats}
            targetLabel={targetLabel}
            streamStatus={streamStatus}
          />
        ) : (
          <CameraPlaceholder 
            onStartCamera={handleStartDetection}
            cameraId={cameraId}
            buttonText={
              initializationStatus.isInitializing 
                ? "Initializing..." 
                : initializationStatus.initializationError 
                  ? "Retry Initialization" 
                  : "Start Optimized Detection"
            }
            icon="detection"
            disabled={
              initializationStatus.isInitializing ||
              !targetLabel || 
              !cameraId || 
              streamStatus.isLoading ||
              (!globalInitializationState.modelLoaded && !initializationStatus.initializationError)
            }
            isLoading={initializationStatus.isInitializing || streamStatus.isLoading}
          />
        )}
      </VideoCard>

      {/* Initialization Status Info */}
      {globalInitializationState.isInitialized && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Chip 
            label="Detection System Ready" 
            size="small" 
            color="success"
            sx={{ mr: 1 }}
          />
          <Chip 
            label={`Model: ${globalInitializationState.modelLoaded ? 'Loaded' : 'Not Loaded'}`} 
            size="small" 
            color={globalInitializationState.modelLoaded ? 'success' : 'warning'}
          />
        </Box>
      )}
    </div>
  );
};

export default DetectionVideoFeed;