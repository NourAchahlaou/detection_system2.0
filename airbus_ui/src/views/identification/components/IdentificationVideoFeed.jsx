// IdentificationVideoFeed.jsx - FIXED: Endless initialization loop + Window Reload on Shutdown
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
import api from "../../../utils/UseAxios";

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
  enableShutdown = true,
  reloadOnShutdown = true // NEW: Option to control window reload
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
  
  // FIXED: Better initialization tracking to prevent loops
  const [initializationStatus, setInitializationStatus] = useState({
    attempted: false,
    inProgress: false,
    completed: false,
    lastAttempt: null
  });
  
  // Health check tracking - FIXED: Better state management
  const [healthCheckStatus, setHealthCheckStatus] = useState({
    initial: false,
    postShutdown: false,
    lastCheck: null,
    inProgress: false
  });

  // Shutdown state tracking
  const [shutdownInProgress, setShutdownInProgress] = useState(false);
  const [shutdownProgress, setShutdownProgress] = useState(null);
  
  const videoRef = useRef(null);
  const mountedRef = useRef(true);
  const stateChangeUnsubscribe = useRef(null);
  const freezeListenerUnsubscribe = useRef(null);
  const healthCheckTimeoutRef = useRef(null);
  
  // FIXED: Single initialization attempt tracking
  const initAttemptRef = useRef({
    attempted: false,
    timestamp: null,
    inProgress: false
  });

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
          // FIXED: Only reset if this is a fresh initialization request
          if (oldState !== IdentificationStates.INITIALIZING) {
            setInitializationStatus({
              attempted: false,
              inProgress: false,
              completed: false,
              lastAttempt: null
            });
            initAttemptRef.current = {
              attempted: false,
              timestamp: null,
              inProgress: false
            };
          }
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
            
            // FIXED: Reset initialization status after shutdown
            setInitializationStatus({
              attempted: false,
              inProgress: false,
              completed: true, // Mark as completed since we're now READY
              lastAttempt: Date.now()
            });
            
            setHealthCheckStatus(prev => ({
              ...prev,
              postShutdown: false,
              inProgress: false
            }));
            
            if (healthCheckTimeoutRef.current) {
              clearTimeout(healthCheckTimeoutRef.current);
              healthCheckTimeoutRef.current = null;
            }
            
          } else if (oldState === IdentificationStates.INITIALIZING) {
            // Initialization completed successfully
            setStreamStatus(prev => ({ ...prev, isLoading: false }));
            setComponentError(null);
            setInitializationStatus(prev => ({
              ...prev,
              inProgress: false,
              completed: true,
              lastAttempt: Date.now()
            }));
            initAttemptRef.current.inProgress = false;
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
          
          if (healthCheckTimeoutRef.current) {
            clearTimeout(healthCheckTimeoutRef.current);
            healthCheckTimeoutRef.current = null;
          }
          setHealthCheckStatus(prev => ({
            ...prev,
            inProgress: false
          }));
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
      if (healthCheckTimeoutRef.current) {
        clearTimeout(healthCheckTimeoutRef.current);
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
      
      setIdentificationStats(prev => ({
        ...prev,
        isFrozen: freezeEvent.status === 'frozen'
      }));
    });
    
    freezeListenerUnsubscribe.current = unsubscribe;

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

  // FIXED: Completely rewritten initialization logic to prevent loops
  useEffect(() => {
    const attemptInitialization = async () => {
      // FIXED: Multiple safeguards to prevent re-initialization
      if (initAttemptRef.current.inProgress) {
        console.log("🚫 Initialization already in progress, skipping...");
        return;
      }
      
      if (initAttemptRef.current.attempted && initializationStatus.completed) {
        console.log("🚫 Initialization already completed successfully, skipping...");
        return;
      }
      
      if (identificationState !== IdentificationStates.INITIALIZING) {
        console.log(`🚫 Not in INITIALIZING state (current: ${identificationState}), skipping...`);
        return;
      }
      
      // FIXED: Prevent rapid re-attempts
      const now = Date.now();
      if (initAttemptRef.current.timestamp && (now - initAttemptRef.current.timestamp) < 5000) {
        console.log("🚫 Too soon since last attempt, waiting...");
        return;
      }
      
      // FIXED: Check if service is already initialized
      if (identificationService.isInitialized && identificationService.getState() === IdentificationStates.READY) {
        console.log("✅ Service already initialized and ready, updating local state...");
        setInitializationStatus({
          attempted: true,
          inProgress: false,
          completed: true,
          lastAttempt: now
        });
        return;
      }
      
      console.log("🚀 IdentificationVideoFeed: Starting initialization...");
      
      // Mark as in progress
      initAttemptRef.current = {
        attempted: true,
        timestamp: now,
        inProgress: true
      };
      
      setInitializationStatus(prev => ({
        ...prev,
        attempted: true,
        inProgress: true,
        lastAttempt: now
      }));
      
      setInitializationAttempts(prev => prev + 1);

      try {
        const initResult = await identificationService.ensureInitialized();
        
        if (!mountedRef.current) return;
        
        if (initResult.success) {
          console.log("✅ IdentificationVideoFeed: Initialization successful");
          setComponentError(null);
          setInitializationStatus(prev => ({
            ...prev,
            inProgress: false,
            completed: true
          }));
          initAttemptRef.current.inProgress = false;
          
          // Schedule health check after successful initialization
          healthCheckTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current && 
                identificationState === IdentificationStates.READY && 
                !healthCheckStatus.inProgress) {
              performInitialHealthCheck();
            }
          }, 2000);
        } else {
          throw new Error(initResult.message || 'Initialization failed');
        }
      } catch (error) {
        console.error("❌ IdentificationVideoFeed: Initialization failed:", error);
        
        if (!mountedRef.current) return;
        
        setComponentError(`Initialization failed: ${error.message}`);
        setInitializationStatus(prev => ({
          ...prev,
          inProgress: false,
          completed: false
        }));
        initAttemptRef.current.inProgress = false;
      }
    };

    // FIXED: Only attempt initialization once when component first loads in INITIALIZING state
    if (identificationState === IdentificationStates.INITIALIZING && 
        !initAttemptRef.current.attempted && 
        !initAttemptRef.current.inProgress) {
      
      // Small delay to prevent immediate execution
      const timeoutId = setTimeout(attemptInitialization, 1000);
      return () => clearTimeout(timeoutId);
    }
  }, [identificationState]); // FIXED: Only depend on identificationState

  // FIXED: Separate effect for health checks that doesn't trigger initialization
  useEffect(() => {
    const scheduleHealthCheck = () => {
      if (identificationState !== IdentificationStates.READY || 
          healthCheckStatus.inProgress ||
          !initializationStatus.completed) {
        return;
      }
      
      const serviceStatus = identificationService.getDetailedStatus();
      
      // Schedule initial health check if needed
      if (!healthCheckStatus.initial && !serviceStatus.hasPerformedInitialHealthCheck) {
        console.log("🩺 Scheduling initial health check...");
        healthCheckTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current && 
              identificationState === IdentificationStates.READY && 
              !healthCheckStatus.inProgress) {
            performInitialHealthCheck();
          }
        }, 2000);
      }
      // Schedule post-shutdown health check if needed
      else if (!healthCheckStatus.postShutdown && 
               !serviceStatus.hasPerformedPostShutdownCheck &&
               !shutdownInProgress) {
        console.log("🩺 Scheduling post-shutdown health check...");
        healthCheckTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current && 
              identificationState === IdentificationStates.READY && 
              !healthCheckStatus.inProgress) {
            performPostShutdownHealthCheck();
          }
        }, 3000);
      }
    };

    const timeoutId = setTimeout(scheduleHealthCheck, 1500);
    return () => clearTimeout(timeoutId);
  }, [
    identificationState, 
    healthCheckStatus.initial, 
    healthCheckStatus.postShutdown, 
    healthCheckStatus.inProgress,
    initializationStatus.completed,
    shutdownInProgress
  ]);

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

  // FIXED: Initial health check with better state management
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckStatus.initial || healthCheckStatus.inProgress) {
      console.log("🚫 Initial health check already completed or in progress");
      return;
    }

    setHealthCheckStatus(prev => ({ ...prev, inProgress: true }));

    try {
      console.log("🩺 IdentificationVideoFeed: Performing initial health check...");
      const health = await identificationService.checkIdentificationHealth(true, false);
      
      if (mountedRef.current) {
        setHealthCheckStatus({
          initial: true,
          postShutdown: healthCheckStatus.postShutdown,
          inProgress: false,
          lastCheck: Date.now()
        });
        
        console.log("✅ IdentificationVideoFeed: Initial health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Initial health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          initial: true,
          inProgress: false,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.initial, healthCheckStatus.inProgress, healthCheckStatus.postShutdown]);

  // FIXED: Post-shutdown health check with better state management
  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckStatus.postShutdown || healthCheckStatus.inProgress) {
      console.log("🚫 Post-shutdown health check already completed or in progress");
      return;
    }

    setHealthCheckStatus(prev => ({ ...prev, inProgress: true }));

    try {
      console.log("🩺 IdentificationVideoFeed: Performing post-shutdown health check...");
      const health = await identificationService.checkIdentificationHealth(false, true);
      
      if (mountedRef.current) {
        setHealthCheckStatus({
          initial: healthCheckStatus.initial,
          postShutdown: true,
          inProgress: false,
          lastCheck: Date.now()
        });
        
        console.log("✅ IdentificationVideoFeed: Post-shutdown health check completed:", health.overall ? "Healthy" : "Issues found");
      }
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Post-shutdown health check failed:", error);
      if (mountedRef.current) {
        setHealthCheckStatus(prev => ({
          ...prev,
          postShutdown: true,
          inProgress: false,
          lastCheck: Date.now()
        }));
      }
    }
  }, [healthCheckStatus.postShutdown, healthCheckStatus.inProgress, healthCheckStatus.initial]);

  // FIXED: Manual retry with complete state reset
  const handleRetryInitialization = useCallback(async () => {
    console.log("🔄 IdentificationVideoFeed: Manual retry requested");
    
    // Clear timeouts
    if (healthCheckTimeoutRef.current) {
      clearTimeout(healthCheckTimeoutRef.current);
      healthCheckTimeoutRef.current = null;
    }
    
    // FIXED: Complete state reset
    setComponentError(null);
    setInitializationAttempts(prev => prev + 1);
    
    // Reset all tracking
    initAttemptRef.current = {
      attempted: false,
      timestamp: null,
      inProgress: false
    };
    
    setInitializationStatus({
      attempted: false,
      inProgress: false,
      completed: false,
      lastAttempt: null
    });
    
    setHealthCheckStatus({
      initial: false,
      postShutdown: false,
      lastCheck: null,
      inProgress: false
    });
    
    try {
      // Force service to reset
      identificationService.resetToInitializing('Manual retry from IdentificationVideoFeed');
      
      // Wait a moment for state to settle
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // The useEffect will handle the actual initialization
      console.log("✅ IdentificationVideoFeed: Retry setup completed, initialization will be handled by useEffect");
      
    } catch (error) {
      console.error("❌ IdentificationVideoFeed: Retry setup failed:", error);
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

  // NEW: Window reload function
  const reloadWindow = useCallback((delay = 2000) => {
    console.log(`🔄 Reloading window in ${delay}ms...`);
    setTimeout(() => {
      window.location.reload();
    }, delay);
  }, []);

  // FIXED: Enhanced shutdown with proper state management + Window Reload
  const performShutdownWithProgress = useCallback(async () => {
    if (shutdownInProgress) {
      console.log("🚫 Shutdown already in progress");
      return;
    }

    setShutdownInProgress(true);
    setShutdownProgress({ message: "Initiating shutdown...", step: 1, total: 4 });

    if (healthCheckTimeoutRef.current) {
      clearTimeout(healthCheckTimeoutRef.current);
      healthCheckTimeoutRef.current = null;
    }

    try {
      console.log("🔥 IdentificationVideoFeed: Starting identification shutdown...");

      if (cameraId) {
        identificationService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
      }

      let shutdownResult;
      
      if (identificationService.canShutdownSafely()) {
        console.log("🛑 Executing complete shutdown with monitoring...");
        shutdownResult = await identificationService.performIdentificationOnlyShutdown();
        
        if (mountedRef.current) {
          setShutdownProgress({ 
            message: "Shutdown completed successfully", 
            step: 4, 
            total: 4 
          });
        }
        
      } else {
        console.log("⚠️ Cannot shutdown safely, stopping streams locally...");
        
        setShutdownProgress({ message: "Stopping local streams...", step: 2, total: 4 });
        await identificationService.stopAllStreamsWithInfrastructure(false);
        
        setShutdownProgress({ message: "Stopping cameras...", step: 3, total: 4 });
        
        try {
          await api.post("/api/artifact_keeper/camera/stop", {}, {
            timeout: 15000,
            headers: { 'Content-Type': 'application/json' }
          });
        } catch (error) {
          console.error('❌ Error stopping cameras:', error);
        }
        
        setShutdownProgress({ message: "Cleaning up resources...", step: 4, total: 4 });
        
        identificationService.resetIdentificationState();
        identificationService.setState('READY', 'Local shutdown completed');
        
        shutdownResult = { success: true, message: 'Local shutdown completed' };
      }

      if (shutdownResult.success && mountedRef.current) {
        console.log("✅ IdentificationVideoFeed: Shutdown completed successfully");
        
        setVideoUrl("");
        resetIdentificationStats();
        setStreamStatus({ isLoading: false, error: null, isConnected: false });
        setComponentError(null);
        setIsStreamFrozen(false);
        setLastIdentificationResult(null);
        setIdentificationInProgress(false);
        setShutdownProgress({ message: "Shutdown complete", step: 4, total: 4 });
        
        // FIXED: Reset all initialization tracking after successful shutdown
        setInitializationAttempts(0);
        initAttemptRef.current = {
          attempted: false,
          timestamp: null,
          inProgress: false
        };
        setInitializationStatus({
          attempted: false,
          inProgress: false,
          completed: false,
          lastAttempt: null
        });
        setHealthCheckStatus({
          initial: false,
          postShutdown: false,
          lastCheck: null,
          inProgress: false
        });
        
        onStopIdentification();

        // NEW: Reload window after shutdown if enabled
        if (reloadOnShutdown) {
          setShutdownProgress({ message: "Reloading page...", step: 4, total: 4 });
          console.log("🔄 Shutdown complete, reloading window...");
          reloadWindow(1500); // Reload after 1.5 seconds
          return; // Exit early since we're reloading
        }

        if (navigateOnStop) {
          navigate('/identification');
        }

        setTimeout(() => {
          if (mountedRef.current) {
            setShutdownProgress(null);
            setShutdownInProgress(false);
          }
        }, 2000);
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
  }, [
    shutdownInProgress, 
    cameraId, 
    handleStatsUpdate, 
    onStopIdentification, 
    navigateOnStop, 
    navigate, 
    resetIdentificationStats,
    reloadOnShutdown,
    reloadWindow
  ]);

  // Enhanced handle stopping identification
  const handleStopIdentification = async () => {
    if (!mountedRef.current) return;
    
    console.log(`🛑 IdentificationVideoFeed: Stopping identification for camera ${cameraId}`);
    
    if (enableShutdown && identificationService.canShutdownSafely()) {
      await performShutdownWithProgress();
    } else {
      setStreamStatus(prev => ({ ...prev, isLoading: true }));

      try {
        identificationService.removeStatsListener(parseInt(cameraId), handleStatsUpdate);
        await identificationService.stopIdentificationStream(parseInt(cameraId));
        
        if (!mountedRef.current) return;
        
        setVideoUrl("");
        resetIdentificationStats();
        setStreamStatus({ isLoading: false, error: null, isConnected: false });
        setComponentError(null);
        setIsStreamFrozen(false);
        setLastIdentificationResult(null);
        setIdentificationInProgress(false);
        
        // FIXED: Reset initialization tracking properly
        setInitializationAttempts(0);
        initAttemptRef.current = {
          attempted: false,
          timestamp: null,
          inProgress: false
        };
        
        onStopIdentification();

        console.log(`✅ IdentificationVideoFeed: Stopped identification for camera ${cameraId}`);

        // NEW: Option to reload after stopping (even without full shutdown)
        if (reloadOnShutdown) {
          console.log("🔄 Stream stopped, reloading window...");
          reloadWindow(1000);
          return;
        }

        if (navigateOnStop) {
          navigate('/identification');
        }

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
    if (identificationInProgress) return;

    setIdentificationInProgress(true);

    try {
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
        
        setIsStreamFrozen(result.streamFrozen);

        setIdentificationStats(prev => ({
          ...prev,
          piecesIdentified: prev.piecesIdentified + (result.summary.total_pieces || 0),
          uniqueLabels: result.summary.unique_labels || prev.uniqueLabels,
          labelCounts: { ...prev.labelCounts, ...result.summary.label_counts },
          lastIdentificationTime: Date.now(),
          avgProcessingTime: result.processingTime,
          isFrozen: result.streamFrozen
        }));
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