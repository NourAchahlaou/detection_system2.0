// hooks/AppIdentificationHooks.js - Updated identification hooks with group support
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

// Custom hook for group management
export const useGroupManagement = () => {
  const [targetGroupName, setTargetGroupName] = useState('');
  const [availableGroups, setAvailableGroups] = useState([]);
  const [currentGroup, setCurrentGroup] = useState(null);
  const [isGroupLoaded, setIsGroupLoaded] = useState(false);
  const [groupLoadingError, setGroupLoadingError] = useState(null);

  // Load available groups on mount
  useEffect(() => {
    const loadGroups = async () => {
      try {
        console.log('📋 Loading available groups...');
        const result = await identificationService.getAvailableGroups();
        
        if (result.success) {
          setAvailableGroups(result.available_groups || []);
          setCurrentGroup(result.current_group || null);
          console.log(`✅ Loaded ${result.available_groups?.length || 0} available groups`);
        } else {
          console.warn('📋 Failed to load available groups:', result.message);
          setAvailableGroups([]);
        }
      } catch (error) {
        console.error('❌ Error loading available groups:', error);
        setAvailableGroups([]);
        setGroupLoadingError(error.message);
      }
    };

    loadGroups();
  }, []);

  // Validate if group is available
  const isGroupAvailable = useCallback((groupName) => {
    if (!groupName || groupName.trim() === '') return false;
    return availableGroups.includes(groupName.trim());
  }, [availableGroups]);

  // Get group validation status
  const getGroupValidationStatus = useCallback(() => {
    if (!targetGroupName || targetGroupName.trim() === '') {
      return { valid: false, message: 'Please enter a group name' };
    }

    if (!isGroupAvailable(targetGroupName)) {
      return { 
        valid: false, 
        message: `Group "${targetGroupName}" not found. Available: ${availableGroups.join(', ')}` 
      };
    }

    return { valid: true, message: `Group "${targetGroupName}" is available` };
  }, [targetGroupName, isGroupAvailable, availableGroups]);

  return {
    // State
    targetGroupName,
    setTargetGroupName,
    availableGroups,
    setAvailableGroups,
    currentGroup,
    setCurrentGroup,
    isGroupLoaded,
    setIsGroupLoaded,
    groupLoadingError,
    setGroupLoadingError,
    
    // Functions
    isGroupAvailable,
    getGroupValidationStatus
  };
};

// Enhanced identification system hook with group support
export const useIdentificationSystem = () => {
  const [identificationState, setIdentificationState] = useState(IdentificationStates.INITIALIZING);
  const [initializationError, setInitializationError] = useState(null);
  const [systemHealth, setSystemHealth] = useState({
    streaming: { status: 'unknown' },
    identification: { status: 'unknown' },
    overall: false
  });
  const [systemProfile, setSystemProfile] = useState(null);
  const [isProfileRefreshing, setIsProfileRefreshing] = useState(false);

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

  // Load system profile
  const loadSystemProfile = useCallback(async () => {
    if (isProfileRefreshing) return;
    
    setIsProfileRefreshing(true);
    try {
      console.log("📊 Loading system profile...");
      const serviceInfo = identificationService.getServiceInfo();
      const detailedStatus = identificationService.getDetailedStatus();
      
      setSystemProfile({
        ...serviceInfo,
        ...detailedStatus,
        timestamp: Date.now()
      });
      
      console.log("✅ System profile loaded successfully");
    } catch (error) {
      console.error("❌ Error loading system profile:", error);
      setSystemProfile(null);
    } finally {
      setIsProfileRefreshing(false);
    }
  }, [isProfileRefreshing]);

  // Load available piece types with group support
  const loadAvailablePieceTypes = useCallback(async (groupName = null) => {
    try {
      let result;
      
      if (groupName) {
        console.log(`📋 Loading piece types for group: ${groupName}`);
        result = await identificationService.getPieceTypesForGroup(groupName);
      } else {
        console.log('📋 Loading general piece types...');
        result = await identificationService.getAvailablePieceTypes();
      }
      
      if (result.success) {
        setAvailablePieceTypes(result.availablePieceTypes || result.available_piece_types || []);
        if (result.confidenceThreshold) {
          setConfidenceThreshold(result.confidenceThreshold);
        }
        console.log(`✅ Loaded ${(result.availablePieceTypes || result.available_piece_types || []).length} piece types`);
      } else {
        console.warn('Failed to load piece types:', result.message);
      }
    } catch (error) {
      console.error("Error loading available piece types:", error);
      setAvailablePieceTypes([]);
    }
  }, []);

  // Enhanced system refresh
  const refreshSystemProfile = useCallback(async () => {
    await loadSystemProfile();
    await performSingleHealthCheck();
  }, [loadSystemProfile, performSingleHealthCheck]);

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
    systemProfile,
    setSystemProfile,
    isProfileRefreshing,
    setIsProfileRefreshing,
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
    loadSystemProfile,
    refreshSystemProfile,
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