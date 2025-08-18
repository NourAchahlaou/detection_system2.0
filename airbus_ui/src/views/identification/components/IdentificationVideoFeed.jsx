// IdentificationVideoFeed.jsx - Enhanced with shutdown integration
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { 
  Box, 
  Alert, 
  CircularProgress, 
  Typography, 
  Button
} from "@mui/material";

import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveIdentificationView from "../LiveIdentificationView";
import { identificationService } from "../service/MainIdentificationService";

// Identification states from service
const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

const IdentificationVideoFeed = ({
  isIdentificationActive,
  onStartIdentification,
  onStopIdentification,
  cameraId,
  confidenceThreshold = 0.5,
  identificationOptions = {},
  navigateOnStop = false,
  enableShutdown = true // New prop to control shutdown functionality
}) => {
  const navigate = useNavigate();
  
  // State management
  const [videoUrl, setVideoUrl] = useState("");
  const [showControls, setShowControls] = useState(false);
  const [identificationStats, setIdentificationStats] = useState({
    piecesIdentified: 0,
    uniqueLabels: 0,
    labelCounts: {},
    lastIdentificationTime: null,
    avgProcessingTime: 0,
    streamQuality: 85,
    queueDepth: 0,
    isStreamActive: false,
    isFrozen: false,
    mode: 'identification'
  });
  const [streamStatus, setStreamStatus] = useState({
    isLoading: false,
    error: null,
    isConnected: false
  });
  
  // Identification service state tracking
  const [identificationState, setIdentificationState] = useState(identificationService.getState());
  const [isModelLoaded, setIsModelLoaded] = useState(identificationService.isModelLoaded);
  const [componentError, setComponentError] = useState(null);
  const [initializationAttempts, setInitializationAttempts] = useState(0);
  
  // Identification specific state
  const [isStreamFrozen, setIsStreamFrozen] = useState(false);
  const [identificationInProgress, setIdentificationInProgress] = useState(false);
  const [lastIdentificationResult, setLastIdentificationResult] = useState(null);
  
  // Health check tracking
  const [healthCheckStatus, setHealthCheckStatus] = useState({
    initial: false,
    postShutdown: false,
    lastCheck: null
  });

  // Shutdown state tracking
  const [shutdownInProgress, setShutdownInProgress] = useState(false);
  const [shutdownProgress, setShutdownProgress] = useState(null);
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);
  const stateChangeUnsubscribe = useRef(null);
  const freezeListenerUnsubscribe = useRef(null);
  const initializationAttempted = useRef(false);

  // Subscribe to identification service state changes
  useEffect(() => {
    mountedRef.current = true;
    
    const unsubscribe = identificationService.addStateChangeListener((newState, oldState) => {
      if (!mountedRef.current) return;
      
      console.log(`🔄 IdentificationVideoFeed: State changed: ${oldState} → ${newState}`);
      setIdentificationState(newState);
      
      // Update model loaded status
      setIsModelLoaded(identificationService.isModelLoaded);
      
      // Handle state-specific logic
      switch (newState) {
        case IdentificationStates.INITIALIZING:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError(null);
          setShutdownInProgress(false);
          setShutdownProgress(null);
          // Reset health check status
          setHealthCheckStatus(prev => ({
            ...prev,
            initial: false,
            postShutdown: false
          }));
          break;
          
        case IdentificationStates.READY:
          if (oldState === IdentificationStates.SHUTTING_DOWN) {
            // Clean up after shutdown
            setStreamStatus({ isLoading: false, error: null, isConnected: false });
            setVideoUrl("");
            setComponentError(null);
            resetIdentificationStats();
            setIsStreamFrozen(false);
            setLastIdentificationResult(null);
            setShutdownInProgress(false);
            setShutdownProgress(null);
            
            // Mark that we need a post-shutdown health check
            setHealthCheckStatus(prev => ({
              ...prev,
              postShutdown: false
            }));
          } else if (oldState === IdentificationStates.INITIALIZING) {
            // Initialization completed
            setStreamStatus(prev => ({ ...prev, isLoading: false }));
            setComponentError(null);
            
            // Mark that we need an initial health check
            setHealthCheckStatus(prev => ({
              ...prev,
              initial: false
            }));
          }
          break;
          
        case IdentificationStates.RUNNING:
          setStreamStatus(prev => ({ ...prev, isLoading: false, isConnected: true }));
          setComponentError(null);
          setShutdownInProgress(false);
          setShutdownProgress(null);
          break;
          
        case IdentificationStates.SHUTTING_DOWN:
          setStreamStatus(prev => ({ ...prev, isLoading: true }));
          setComponentError("System is shutting down...");
          setIsStreamFrozen(false);
          break;
      }
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
    
    // Get initial state and model status
    setIdentificationState(identificationService.getState());
    setIsModelLoaded(identificationService.isModelLoaded);
    
    return () => {
      mountedRef.current = false;
      if (stateChangeUnsubscribe.current) {
        stateChangeUnsubscribe.current();
      }
    };
  }, []);

  // Subscribe to freeze/unfreeze events
  useEffect(() => {
    if (!cameraId) return;
    
    const unsubscribe = identificationService.addFreezeListener((freezeEvent) => {
      if (!mountedRef.current || freezeEvent.cameraId !== parseInt(cameraId)) return;
      
      console.log(`🧊 IdentificationVideoFeed: Freeze event for camera ${cameraId}:`, freezeEvent);
      setIsStreamFrozen(freezeEvent.status === 'frozen');
      
      // Update identification stats
      setIdentificationStats(prev => ({
        ...prev,
        isFrozen: freezeEvent.status === 'frozen'
      }));
    });
    
    freezeListenerUnsubscribe.current = unsubscribe;

    // Check initial freeze status
    if (identificationService.isStreamFrozen(cameraId)) {
      setIsStreamFrozen(true);
      setIdentificationStats(prev => ({ ...prev, isFrozen: true }));
    }
    
    return () => {
      if (freezeListenerUnsubscribe.current) {
        freezeListenerUnsubscribe.current();
      }
    };
  }, [cameraId]);

  // Initialize identification system on component mount
  useEffect(() => {
    const initializeIfNeeded = async () => {
      if (identificationState === IdentificationStates.INITIALIZING && !initializationAttempted.current) {
        console.log("🚀 IdentificationVideoFeed: Starting identification system initialization...");
        initializationAttempted.current = true;
        setInitializationAttempts(1);
        
        try {
          const initResult = await identificationService.ensureInitialized();
          
          if (initResult.success && mountedRef.current) {
            console.log("✅ IdentificationVideoFeed: Identification system initialized successfully");
            setComponentError(null);
            
            setTimeout(() => {
              if (mountedRef.current && identificationState === IdentificationStates.READY) {
                performInitialHealthCheck();
              }
            }, 1000);
          }
        } catch (error) {
          console.error("❌ IdentificationVideoFeed: Initialization failed:", error);
          if (mountedRef.current) {
            setComponentError(`Initialization failed: ${error.message}`);
          }
        }
      }
    };

    initializeIfNeeded();
  }, [identificationState]);

  // Watch for READY state transitions to trigger appropriate health checks
  useEffect(() => {
    const handleReadyStateTransition = async () => {
      if (identificationState !== IdentificationStates.READY) return;
      
      const serviceStatus = identificationService.getDetailedStatus();
      
      if (!healthCheckStatus.initial && !serviceStatus.hasPerformedInitialHealthCheck) {
        console.log("🩺 IdentificationVideoFeed: Triggering initial health check...");
        await performInitialHealthCheck();
      }
      else if (!healthCheckStatus.postShutdown && !serviceStatus.hasPerformedPostShutdownCheck) {
        console.log("🩺 IdentificationVideoFeed: Triggering post-shutdown health check...");
        await performPostShutdownHealthCheck();
      }
    };

    const timer = setTimeout(handleReadyStateTransition, 500);
    return () => clearTimeout(timer);
  }, [identificationState, healthCheckStatus.initial, healthCheckStatus.postShutdown]);

  // Reset identification stats helper
  const resetIdentificationStats = useCallback(() => {
    setIdentificationStats({
      piecesIdentified: 0,
      uniqueLabels: 0,
      labelCounts: {},
      lastIdentificationTime: null,
      avgProcessingTime: 0,
      streamQuality: 85,
      queueDepth: 0,
      isStreamActive: false,
      isFrozen: false,
      mode: 'identification'
    });
    setLastIdentificationResult(null);
  }, []);

  // Initial health check function
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckStatus.initial) return;

    try {
      console.log("🩺 IdentificationVideoFeed: Performing initial health check...");
      const health = await identificationService.checkIdentificationHealth(true, false);
      
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          initial: true,
          lastCheck: Date.now()
        }));
        
        console.log("✅ IdentificationVideoFeed: Initial health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Initial health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          initial: true,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.initial]);

  // Post-shutdown health check function
  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckStatus.postShutdown) return;

    try {
      console.log("🩺 IdentificationVideoFeed: Performing post-shutdown health check...");
      const health = await identificationService.checkIdentificationHealth(false, true);
      
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          postShutdown: true,
          lastCheck: Date.now()
        }));
        
        console.log("✅ IdentificationVideoFeed: Post-shutdown health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Post-shutdown health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          postShutdown: true,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.postShutdown]);

  // Manual retry initialization
  const handleRetryInitialization = useCallback(async () => {
    console.log("🔄 IdentificationVideoFeed: Manual initialization retry requested");
    setComponentError(null);
    setInitializationAttempts(prev => prev + 1);
    initializationAttempted.current = false;
    
    setHealthCheckStatus({
      initial: false,
      postShutdown: false,
      lastCheck: null
    });
    
    try {
      identificationService.resetToInitializing('Manual retry from IdentificationVideoFeed');
      await new Promise(resolve => setTimeout(resolve, 500));
      const initResult = await identificationService.ensureInitialized();
      
      if (initResult.success && mountedRef.current) {
        console.log("✅ IdentificationVideoFeed: Retry initialization successful");
        setComponentError(null);
      }
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Retry initialization failed:", error);
      if (mountedRef.current) {
        setComponentError(`Retry failed: ${error.message}`);
      }
    }
  }, []);

  // Stats listener callback
  const handleStatsUpdate = useCallback((newStats) => {
    if (!mountedRef.current || identificationState === IdentificationStates.SHUTTING_DOWN) return;
    
    setIdentificationStats(prevStats => ({
      ...prevStats,
      ...newStats,
      mode: 'identification',
      piecesIdentified: newStats.piecesIdentified || prevStats.piecesIdentified,
      uniqueLabels: newStats.uniqueLabels || prevStats.uniqueLabels,
      labelCounts: newStats.labelCounts || prevStats.labelCounts
    }));
  }, [identificationState]);

  // Enhanced identification start
  const handleStartIdentification = async () => {
    if (!cameraId || cameraId === '') {
      setComponentError("Please select a camera first.");
      return;
    }

    console.log(`🎯 IdentificationVideoFeed: Attempting to start identification. Current state: ${identificationState}`);

    const currentState = identificationService.getState();
    
    if (currentState === IdentificationStates.INITIALIZING) {
      setComponentError("System is still initializing. Please wait...");
      return;
    }

    if (currentState === IdentificationStates.SHUTTING_DOWN) {
      setComponentError("System is shutting down. Please wait for it to complete.");
      return;
    }

    if (currentState !== IdentificationStates.READY) {
      setComponentError(`Cannot start identification. Current state: ${currentState}. System must be READY.`);
      return;
    }

    if (!identificationService.isModelLoaded) {
      setComponentError("Identification model is not loaded. Please wait or try refreshing.");
      return;
    }

    setStreamStatus({ isLoading: true, error: null, isConnected: false });
    setComponentError(null);

    try {
      console.log(`🎯 IdentificationVideoFeed: Starting identification stream for camera ${cameraId}`);
      
      const streamUrl = await identificationService.startIdentificationStream(
        parseInt(cameraId), 
        {
          streamQuality: identificationOptions.streamQuality || 85,
          confidenceThreshold: confidenceThreshold,
          priority: identificationOptions.priority || 1
        }
      );

      if (!mountedRef.current) return;

      setVideoUrl(streamUrl);
      
      // Add stats listener
      identificationService.addStatsListener(parseInt(cameraId), handleStatsUpdate);
      
      setIdentificationStats({
        piecesIdentified: 0,
        uniqueLabels: 0,
        labelCounts: {},
        lastIdentificationTime: null,
        avgProcessingTime: 0,
        streamQuality: identificationOptions.streamQuality || 85,
        queueDepth: 0,
        isStreamActive: true,
        isFrozen: false,
        mode: 'identification'
      });

      setStreamStatus({ isLoading: false, error: null, isConnected: true });
      onStartIdentification();

      console.log(`✅ IdentificationVideoFeed: Started identification for camera ${cameraId}`);

    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Error starting identification:", error);
      
      if (!mountedRef.current) return;
      
      setStreamStatus({ 
        isLoading: false, 
        error: error.message || "Failed to start identification", 
        isConnected: false 
      });
      setComponentError(`Failed to start identification: ${error.message}`);
    }
  };

// Enhanced shutdown functionality with monitoring - FIXED VERSION
const performShutdownWithProgress = useCallback(async () => {
  if (shutdownInProgress) {
    console.log("🚫 Shutdown already in progress");
    return;
  }

  setShutdownInProgress(true);
  setShutdownProgress({ message: "Initiating shutdown...", step: 1, total: 4 });

  try {
    console.log("🔥 IdentificationVideoFeed: Starting identification shutdown...");

    // Remove stats listener first
    if (cameraId) {
      identificationService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
    }

    // FIXED: Actually perform the shutdown instead of just monitoring
    let shutdownResult;
    
    if (identificationService.canShutdownSafely()) {
      // Monitor progress while performing shutdown
      const progressCallback = (progress) => {
        if (mountedRef.current) {
          setShutdownProgress({
            message: progress.message || "Shutting down...",
            step: progress.step || 1,
            total: progress.total || 4,
            details: progress.details
          });
        }
      };

      // FIXED: Use executeShutdown with identification-only option
      console.log("🛑 Executing identification-only shutdown with monitoring...");
      shutdownResult = await identificationService.executeShutdown('identification_only', true);
      
      // Update progress during shutdown
      if (mountedRef.current) {
        setShutdownProgress({ 
          message: "Shutdown completed successfully", 
          step: 4, 
          total: 4 
        });
      }
      
    } else {
      // Fallback: Stop streams without backend shutdown
      console.log("⚠️ Cannot shutdown safely, stopping streams locally...");
      
      setShutdownProgress({ message: "Stopping local streams...", step: 2, total: 4 });
      
      // Stop all streams with infrastructure cleanup
      await identificationService.stopAllStreamsWithInfrastructure(false);
      
      setShutdownProgress({ message: "Cleaning up resources...", step: 3, total: 4 });
      
      // Reset identification state
      identificationService.resetIdentificationState();
      identificationService.setState('READY', 'Local shutdown completed');
      
      shutdownResult = {
        success: true,
        message: 'Local shutdown completed - backend services remain running',
        type: 'local_only'
      };
    }

    if (shutdownResult.success) {
      console.log("✅ IdentificationVideoFeed: Shutdown completed successfully");
      
      if (mountedRef.current) {
        setVideoUrl("");
        resetIdentificationStats();
        setStreamStatus({ isLoading: false, error: null, isConnected: false });
        setComponentError(null);
        setIsStreamFrozen(false);
        setLastIdentificationResult(null);
        setIdentificationInProgress(false);
        setShutdownProgress({ message: "Shutdown complete", step: 4, total: 4 });
        
        // Call the parent's onStopIdentification callback
        onStopIdentification();

        // Navigate to /identification route if enabled
        if (navigateOnStop) {
          console.log("🧭 IdentificationVideoFeed: Navigating to /identification route");
          navigate('/identification');
        }

        // Clear shutdown progress after a delay
        setTimeout(() => {
          if (mountedRef.current) {
            setShutdownProgress(null);
            setShutdownInProgress(false);
          }
        }, 2000);
      }
    } else {
      throw new Error(shutdownResult.message || 'Shutdown failed');
    }

  } catch (error) {
    console.error("❌ IdentificationVideoFeed: Error during shutdown:", error);
    
    if (mountedRef.current) {
      setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to shutdown properly" }));
      setComponentError(`Shutdown error: ${error.message}`);
      setShutdownProgress(null);
      setShutdownInProgress(false);
    }
  }
}, [cameraId, shutdownInProgress, handleStatsUpdate, onStopIdentification, navigateOnStop, navigate, resetIdentificationStats]);
  // Enhanced handle stopping identification with shutdown integration
  const handleStopIdentification = async () => {
    if (!mountedRef.current) return;
    
    console.log(`🛑 IdentificationVideoFeed: Stopping identification for camera ${cameraId}`);
    
    if (enableShutdown && identificationService.canShutdownSafely()) {
      // Use enhanced shutdown with progress monitoring
      await performShutdownWithProgress();
    } else {
      // Fallback to simple stop if shutdown is disabled or not safe
      setStreamStatus(prev => ({ ...prev, isLoading: true }));

      try {
        // Remove stats listener first
        identificationService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
        
        // Stop identification stream
        await identificationService.stopIdentificationStream(parseInt(cameraId));
        
        if (!mountedRef.current) return;
        
        setVideoUrl("");
        resetIdentificationStats();
        setStreamStatus({ isLoading: false, error: null, isConnected: false });
        setComponentError(null);
        setIsStreamFrozen(false);
        setLastIdentificationResult(null);
        setIdentificationInProgress(false);
        
        // Call the parent's onStopIdentification callback
        onStopIdentification();

        console.log(`✅ IdentificationVideoFeed: Stopped identification for camera ${cameraId}`);

        // Navigate to /identification route if enabled
        if (navigateOnStop) {
          console.log("🧭 IdentificationVideoFeed: Navigating to /identification route");
          navigate('/identification');
        }

        // Schedule post-shutdown health check
        setTimeout(() => {
          if (mountedRef.current && identificationState === IdentificationStates.READY) {
            console.log("🩺 IdentificationVideoFeed: Scheduling post-shutdown health check...");
            setHealthCheckStatus(prev => ({
              ...prev,
              postShutdown: false
            }));
            performPostShutdownHealthCheck();
          }
        }, 2000);

      } catch (error) {
        console.error("❌ IdentificationVideoFeed: Error stopping identification:", error);
        
        if (!mountedRef.current) return;
        
        setStreamStatus(prev => ({ ...prev, isLoading: false, error: "Failed to stop identification" }));
        setComponentError(`Failed to stop identification: ${error.message}`);
      }
    }
  };

  // Piece identification handler
  const handlePieceIdentification = async () => {
    if (identificationInProgress) {
      console.log("🚫 Identification already in progress, skipping...");
      return;
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing piece identification for camera ${cameraId}`);
      
      const result = await identificationService.performPieceIdentification(cameraId, {
        freezeStream: true,
        quality: identificationOptions.streamQuality || 85,
        confidenceThreshold: confidenceThreshold
      });

      if (result.success && mountedRef.current) {
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

        // Update stats
        setIdentificationStats(prev => ({
          ...prev,
          piecesIdentified: prev.piecesIdentified + (result.summary.total_pieces || 0),
          uniqueLabels: result.summary.unique_labels || prev.uniqueLabels,
          labelCounts: { ...prev.labelCounts, ...result.summary.label_counts },
          lastIdentificationTime: Date.now(),
          avgProcessingTime: result.processingTime,
          isFrozen: result.streamFrozen
        }));

        console.log(`✅ Piece identification completed: ${result.summary.total_pieces} pieces found`);
      }
    } catch (error) {
      console.error('❌ Error in piece identification:', error);
      setComponentError(`Identification failed: ${error.message}`);
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Quick analysis handler
  const handleQuickAnalysis = async () => {
    if (identificationInProgress) {
      console.log("🚫 Analysis already in progress, skipping...");
      return;
    }

    setIdentificationInProgress(true);

    try {
      console.log(`🔍 Performing quick analysis for camera ${cameraId}`);
      
      const result = await identificationService.performQuickAnalysis(cameraId, {
        analyzeFrameOnly: true,
        quality: identificationOptions.streamQuality || 85,
        confidenceThreshold: confidenceThreshold
      });

      if (result.success && mountedRef.current) {
        setLastIdentificationResult({
          piecesFound: result.piecesFound,
          pieces: result.pieces,
          summary: result.summary,
          processingTime: result.processingTime,
          timestamp: result.timestamp,
          message: result.message,
          isQuickAnalysis: true
        });

        console.log(`✅ Quick analysis completed: ${result.piecesFound} pieces found`);
      }
    } catch (error) {
      console.error('❌ Error in quick analysis:', error);
      setComponentError(`Quick analysis failed: ${error.message}`);
    } finally {
      setIdentificationInProgress(false);
    }
  };

  // Freeze/unfreeze stream handlers
  const handleFreezeStream = async () => {
    try {
      await identificationService.freezeStream(cameraId);
    } catch (error) {
      console.error("❌ Error freezing stream:", error);
      setComponentError(`Failed to freeze stream: ${error.message}`);
    }
  };

  const handleUnfreezeStream = async () => {
    try {
      await identificationService.unfreezeStream(cameraId);
    } catch (error) {
      console.error("❌ Error unfreezing stream:", error);
      setComponentError(`Failed to unfreeze stream: ${error.message}`);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraId) {
        identificationService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }
    };
  }, [cameraId, handleStatsUpdate]);

  // Get appropriate button text based on state
  const getButtonText = () => {
    switch (identificationState) {
      case IdentificationStates.INITIALIZING:
        return "Initializing System...";
      case IdentificationStates.READY:
        return componentError ? "Retry" : "Start Identification";
      case IdentificationStates.RUNNING:
        return "System Running";
      case IdentificationStates.SHUTTING_DOWN:
        return shutdownProgress ? shutdownProgress.message : "Shutting Down...";
      default:
        return "Start Stream";
    }
  };

  // Determine if button should be disabled
  const isButtonDisabled = () => {
    return (
      identificationState === IdentificationStates.INITIALIZING ||
      identificationState === IdentificationStates.SHUTTING_DOWN ||
      (identificationState === IdentificationStates.RUNNING && isIdentificationActive) ||
      !cameraId || 
      streamStatus.isLoading ||
      (!isModelLoaded && identificationState !== IdentificationStates.INITIALIZING) ||
      shutdownInProgress
    );
  };

  // Get appropriate loading state
  const isLoading = () => {
    return (
      identificationState === IdentificationStates.INITIALIZING ||
      identificationState === IdentificationStates.SHUTTING_DOWN ||
      streamStatus.isLoading ||
      shutdownInProgress
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* State-based Status Alerts */}
      {identificationState === IdentificationStates.INITIALIZING && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Initializing identification system... Please wait.
        </Alert>
      )}

      {identificationState === IdentificationStates.SHUTTING_DOWN && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {shutdownProgress ? 
            `${shutdownProgress.message} (${shutdownProgress.step}/${shutdownProgress.total})` :
            "System is shutting down... This may take a moment."
          }
          {shutdownProgress?.details && (
            <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
              {shutdownProgress.details}
            </Typography>
          )}
        </Alert>
      )}

      {componentError && (
        <Alert 
          severity="error" 
          sx={{ mb: 2, width: '100%' }}
          action={
            identificationState === IdentificationStates.INITIALIZING ? (
              <Button 
                color="inherit" 
                size="small" 
                onClick={handleRetryInitialization}
                disabled={initializationAttempts >= 3}
              >
                {initializationAttempts >= 3 ? 'Max Retries' : 'Retry'}
              </Button>
            ) : null
          }
        >
          {componentError}
          {initializationAttempts >= 3 && (
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Maximum retry attempts reached. Please refresh the page.
            </Typography>
          )}
        </Alert>
      )}

      {/* Stream Status Alerts */}
      {streamStatus.error && (
        <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
          {streamStatus.error}
        </Alert>
      )}

      {streamStatus.isLoading && identificationState !== IdentificationStates.SHUTTING_DOWN && !shutdownInProgress && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          {isIdentificationActive ? "Connecting to identification stream..." : "Stopping stream..."}
        </Alert>
      )}

      {/* Shutdown Progress Alert */}
      {shutdownInProgress && shutdownProgress && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          <Box>
            <Typography variant="body2">
              {shutdownProgress.message} ({shutdownProgress.step}/{shutdownProgress.total})
            </Typography>
          </Box>
        </Alert>
      )}

      {/* Identification Specific Alerts */}
      {isStreamFrozen && (
        <Alert severity="info" sx={{ mb: 2, width: '100%' }}>
          <Typography variant="body2">
            Stream is frozen for identification. Use controls below to unfreeze or perform identification.
          </Typography>
        </Alert>
      )}

      {identificationInProgress && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, width: '100%', display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Performing piece identification...
        </Alert>
      )}

      <VideoCard
        cameraActive={isIdentificationActive && identificationState === IdentificationStates.RUNNING}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isIdentificationActive && identificationState === IdentificationStates.RUNNING ? (
          <LiveIdentificationView
            videoUrl={videoUrl}
            videoRef={videoRef}
            showControls={showControls}
            onStopIdentification={handleStopIdentification}
            identificationStats={identificationStats}
            confidenceThreshold={confidenceThreshold}
            streamStatus={streamStatus}
            isStreamFrozen={isStreamFrozen}
            lastIdentificationResult={lastIdentificationResult}
            identificationInProgress={identificationInProgress}
            onPieceIdentification={handlePieceIdentification}
            onQuickAnalysis={handleQuickAnalysis}
            onFreezeStream={handleFreezeStream}
            onUnfreezeStream={handleUnfreezeStream}
          />
        ) : (
          <CameraPlaceholder 
            onStartCamera={componentError && identificationState === IdentificationStates.INITIALIZING 
              ? handleRetryInitialization 
              : handleStartIdentification}
            cameraId={cameraId}
            buttonText={getButtonText()}
            icon="identification"
            disabled={isButtonDisabled()}
            isLoading={isLoading()}
          />
        )}
      </VideoCard>
    </div>
  );
};

export default IdentificationVideoFeed;