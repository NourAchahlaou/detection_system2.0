// hooks/AppIdentificationHooks.js - Simplified identification hooks without workflows
import { useState, useEffect, useCallback, useRef } from "react";
import { cameraService } from "../../captureImage/CameraService";
import { identificationService } from "../service/MainIdentificationService";

// Identification states from service
const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export const useIdentificationManagement = () => {
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const showSnackbar = useCallback((message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  }, []);
  return {

    snackbar,
    setSnackbar,
    showSnackbar,

  };
};
// Custom hook for identification system management
export const useIdentificationSystem = () => {
  const [identificationState, setIdentificationState] = useState(IdentificationStates.INITIALIZING);
  const [initializationError, setInitializationError] = useState(null);
  const [systemHealth, setSystemHealth] = useState({
    streaming: { status: 'unknown' },
    identification: { status: 'unknown' },
    overall: false
  });

  // Identification specific state
  const [isStreamFrozen, setIsStreamFrozen] = useState(false);
  const [identificationInProgress, setIdentificationInProgress] = useState(false);
  const [lastIdentificationResult, setLastIdentificationResult] = useState(null);
  const [availablePieceTypes, setAvailablePieceTypes] = useState([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  // Refs for lifecycle management
  const cleanupRef = useRef(false);
  const initializationAttempted = useRef(false);
  const lastHealthCheck = useRef(null);
  const stateChangeUnsubscribe = useRef(null);
  const freezeListenerUnsubscribe = useRef(null);
  const healthCheckPerformed = useRef({
    initial: false,
    postShutdown: false
  });

  // Subscribe to identification service state changes
  useEffect(() => {
    const unsubscribe = identificationService.addStateChangeListener((newState, oldState) => {
      console.log(`🔄 Identification state changed: ${oldState} → ${newState}`);
      setIdentificationState(newState);
      
      // Reset health check flags on state transitions
      if (newState === IdentificationStates.INITIALIZING) {
        healthCheckPerformed.current.initial = false;
        healthCheckPerformed.current.postShutdown = false;
      } else if (newState === IdentificationStates.READY && oldState === IdentificationStates.SHUTTING_DOWN) {
        healthCheckPerformed.current.postShutdown = false;
      }
    });
    
    stateChangeUnsubscribe.current = unsubscribe;
    setIdentificationState(identificationService.getState());
    
    return () => {
      if (stateChangeUnsubscribe.current) {
        stateChangeUnsubscribe.current();
      }
    };
  }, []);

  // Subscribe to freeze/unfreeze events
  useEffect(() => {
    const unsubscribe = identificationService.addFreezeListener((freezeEvent) => {
      console.log(`🧊 Freeze event:`, freezeEvent);
      setIsStreamFrozen(freezeEvent.status === 'frozen');
    });
    
    freezeListenerUnsubscribe.current = unsubscribe;
    
    return () => {
      if (freezeListenerUnsubscribe.current) {
        freezeListenerUnsubscribe.current();
      }
    };
  }, []);

  // Health check functions
  const performInitialHealthCheck = useCallback(async () => {
    if (healthCheckPerformed.current.initial) {
      console.log("⏭️ Initial health check already performed");
      return;
    }

    if (identificationState === IdentificationStates.SHUTTING_DOWN) {
      console.log("⏭️ Skipping initial health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing initial health check...");
      const health = await identificationService.checkIdentificationHealth(true, false);
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.initial = true;
      
      console.log("✅ Initial health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Initial health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        identification: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.initial = true;
    }
  }, [identificationState]);

  const performPostShutdownHealthCheck = useCallback(async () => {
    if (healthCheckPerformed.current.postShutdown) {
      console.log("⏭️ Post-shutdown health check already performed");
      return;
    }

    try {
      console.log("🩺 Performing post-shutdown health check...");
      const health = await identificationService.checkIdentificationHealth(false, true);
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.postShutdown = true;
      
      console.log("✅ Post-shutdown health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Post-shutdown health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        identification: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
      healthCheckPerformed.current.postShutdown = true;
    }
  }, []);

  const performSingleHealthCheck = useCallback(async () => {
    if (identificationState === IdentificationStates.SHUTTING_DOWN) {
      console.log("⏭️ Skipping health check - system is shutting down");
      return;
    }

    try {
      console.log("🩺 Performing manual health check...");
      const health = await identificationService.checkIdentificationHealth();
      setSystemHealth(health);
      lastHealthCheck.current = Date.now();
      
      if (!health.overall) {
        console.warn("System health check failed:", health);
      }
      
      console.log("✅ Manual health check completed:", health.overall ? "Healthy" : "Issues found");
    } catch (error) {
      console.error("Manual health check error:", error);
      setSystemHealth({
        streaming: { status: 'unhealthy', error: error.message },
        identification: { status: 'unhealthy', error: error.message },
        overall: false
      });
      lastHealthCheck.current = Date.now();
    }
  }, [identificationState]);

  // Load available piece types
  const loadAvailablePieceTypes = useCallback(async () => {
    try {
      const result = await identificationService.getAvailablePieceTypes();
      if (result.success) {
        setAvailablePieceTypes(result.availablePieceTypes || []);
        setConfidenceThreshold(result.confidenceThreshold || 0.5);
      }
    } catch (error) {
      console.error("Error loading available piece types:", error);
    }
  }, []);

  // Helper functions
  const getHealthCheckAge = () => {
    if (!lastHealthCheck.current) return 'Never';
    const ageMs = Date.now() - lastHealthCheck.current;
    const ageSeconds = Math.floor(ageMs / 1000);
    if (ageSeconds < 60) return `${ageSeconds}s ago`;
    const ageMinutes = Math.floor(ageSeconds / 60);
    return `${ageMinutes}m ago`;
  };

  return {
    // State
    identificationState,
    initializationError,
    setInitializationError,
    systemHealth,
    isStreamFrozen,
    setIsStreamFrozen,
    identificationInProgress,
    setIdentificationInProgress,
    lastIdentificationResult,
    setLastIdentificationResult,
    availablePieceTypes,
    setAvailablePieceTypes,
    confidenceThreshold,
    setConfidenceThreshold,
    
    // Refs
    initializationAttempted,
    healthCheckPerformed,
    lastHealthCheck,
    cleanupRef,
    
    // Functions
    performInitialHealthCheck,
    performPostShutdownHealthCheck,
    performSingleHealthCheck,
    loadAvailablePieceTypes,
    getHealthCheckAge,
    
    // Constants
    IdentificationStates
  };
};

// Custom hook for camera management (reused from detection)
export const useCameraManagement = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);

  // Fetch available cameras
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const cameraData = await cameraService.getAllCameras();
        setCameras(cameraData);
        console.log(`Loaded ${cameraData.length} cameras`);
      } catch (error) {
        console.error("Error fetching cameras:", error);
        setCameras([]);
      }
    };
    
    fetchCameras();
  }, []);

  // Enhanced camera detection
  const handleDetectCameras = useCallback(async (showSnackbar) => {
    setIsDetecting(true);
    try {
      console.log("Detecting available cameras...");
      const detectedCameras = await cameraService.detectCameras();
      setCameras(detectedCameras);
      
      if (selectedCameraId && !detectedCameras.some(cam => cam.id.toString() === selectedCameraId.toString())) {
        console.log("Previously selected camera no longer available, resetting selection");
        setSelectedCameraId('');
        setCameraId('');
        
        if (showSnackbar) {
          showSnackbar("The camera currently in use is no longer available. Identification has been stopped.", "warning");
        }
      }
      
      console.log(`Successfully detected ${detectedCameras.length} cameras`);
    } catch (error) {
      console.error("Error detecting cameras:", error);
      if (showSnackbar) {
        showSnackbar(`Camera detection failed: ${error.message}`, "error");
      }
    } finally {
      setIsDetecting(false);
    }
  }, [selectedCameraId]);

  return {
    cameras,
    setCameras,
    selectedCameraId,
    setSelectedCameraId,
    cameraId,
    setCameraId,
    isDetecting,
    handleDetectCameras
  };
};