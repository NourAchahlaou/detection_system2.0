import api from "../../utils/UseAxios";

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

class DetectionService {
  constructor() {
    this.currentStreams = new Map();
    this.detectionStats = new Map();
    this.wsConnections = new Map();
    this.eventListeners = new Map();
    
    // Clear state management with proper initialization
    this.state = DetectionStates.INITIALIZING;
    this.isModelLoaded = false;
    this.lastHealthCheck = null;
    this.healthCheckInProgress = false;
    this.stateChangeListeners = new Set();
    
    // Initialization tracking
    this.initializationPromise = null;
    this.hasPerformedInitialHealthCheck = false; // Track if initial health check was done
    this.hasPerformedPostShutdownCheck = false; // Track post-shutdown check
    
    // Timeouts
    this.INITIALIZATION_TIMEOUT = 30000;
    this.HEALTH_CHECK_TIMEOUT = 10000;
    this.CAMERA_START_TIMEOUT = 15000;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.HEALTH_CHECK_COOLDOWN = 5000;
    
    console.log('🔧 DetectionService initialized with state:', this.state);
  }

  // State management methods
  setState(newState, reason = '') {
    const oldState = this.state;
    this.state = newState;
    console.log(`🔄 State changed: ${oldState} → ${newState}${reason ? ` (${reason})` : ''}`);
    
    // Reset health check flags on state transitions
    if (newState === DetectionStates.INITIALIZING) {
      this.hasPerformedInitialHealthCheck = false;
      this.hasPerformedPostShutdownCheck = false;
    } else if (newState === DetectionStates.READY && oldState === DetectionStates.SHUTTING_DOWN) {
      this.hasPerformedPostShutdownCheck = false; // Allow post-shutdown check
    }
    
    // Notify all listeners
    this.stateChangeListeners.forEach(listener => {
      try {
        listener(newState, oldState);
      } catch (error) {
        console.error('Error in state change listener:', error);
      }
    });
  }

  getState() {
    return this.state;
  }

  addStateChangeListener(listener) {
    this.stateChangeListeners.add(listener);
    return () => this.stateChangeListeners.delete(listener);
  }

  // Check if operations are allowed in current state
  canInitialize() {
    return this.state === DetectionStates.INITIALIZING;
  }

  canStart() {
    return this.state === DetectionStates.READY;
  }

  canStop() {
    return this.state === DetectionStates.RUNNING;
  }

  canShutdown() {
    return [DetectionStates.READY, DetectionStates.RUNNING].includes(this.state);
  }

  isOperational() {
    return [DetectionStates.READY, DetectionStates.RUNNING].includes(this.state);
  }

  resetToInitializing(reason = 'Manual reset') {
    this.setState(DetectionStates.INITIALIZING, reason);
    this.isModelLoaded = false;
    this.initializationPromise = null;
    this.lastHealthCheck = null;
    this.healthCheckInProgress = false;
    this.hasPerformedInitialHealthCheck = false;
    this.hasPerformedPostShutdownCheck = false;
    console.log('🔄 Detection service reset to initializing state');
  }

  // Updated shouldSkipHealthCheck with clearer logic
  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    // NEVER skip initial health checks during initialization
    if (isInitialCheck && !this.hasPerformedInitialHealthCheck) {
      console.log('🩺 Allowing initial health check during initialization');
      return false;
    }

    // NEVER skip post-shutdown health checks
    if (isPostShutdownCheck && !this.hasPerformedPostShutdownCheck) {
      console.log('🩺 Allowing post-shutdown health check');
      return false;
    }

    // Skip if system is shutting down (except for post-shutdown checks)
    if (this.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏭️ Skipping health check - system is shutting down');
      return true;
    }

    // Skip if another health check is in progress
    if (this.healthCheckInProgress) {
      console.log('⏭️ Skipping health check - already in progress');
      return true;
    }

    // Skip if too soon since last check (cooldown period)
    if (this.lastHealthCheck && (Date.now() - this.lastHealthCheck) < this.HEALTH_CHECK_COOLDOWN) {
      console.log('⏭️ Skipping health check - too soon since last check');
      return true;
    }

    return false;
  }

  async ensureInitialized() {
    // If already ready or running, no need to initialize
    if (this.isOperational() && this.isModelLoaded) {
      return { success: true, message: 'Already initialized', state: this.state };
    }

    // If currently shutting down, wait for it to complete then reset
    if (this.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏳ Waiting for shutdown to complete before initializing...');
      // Wait a bit for shutdown to complete
      await new Promise(resolve => setTimeout(resolve, 2000));
      this.resetToInitializing('Post-shutdown reset');
    }

    // If not in initializing state, reset to it
    if (this.state !== DetectionStates.INITIALIZING) {
      this.resetToInitializing('Ensure initialization');
    }

    // If initialization is already in progress, wait for it
    if (this.initializationPromise) {
      try {
        console.log('⏳ Waiting for existing initialization to complete...');
        return await Promise.race([
          this.initializationPromise,
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Initialization timeout')), this.INITIALIZATION_TIMEOUT)
          )
        ]);
      } catch (error) {
        console.error('❌ Existing initialization failed or timed out:', error.message);
        this.resetToInitializing('Failed initialization cleanup');
        throw error;
      }
    }

    // Start new initialization
    this.initializationPromise = this.initializeProcessor();
    return await this.initializationPromise;
  }

  async initializeProcessor() {
    if (!this.canInitialize()) {
      throw new Error(`Cannot initialize from state: ${this.state}`);
    }

    console.log('🚀 Starting detection processor initialization...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.INITIALIZATION_TIMEOUT);

      const response = await api.post('/api/detection/redis/initialize', {}, {
        signal: controller.signal,
        timeout: this.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running') {
        console.log('✅ Detection processor initialized:', response.data.message);
        
        // Load the model with initial health check
        const modelResult = await this.loadModel(true); // Pass true for initial check
        if (modelResult.success) {
          this.isModelLoaded = true;
          this.setState(DetectionStates.READY, 'Initialization completed');
          
          return {
            success: true,
            message: 'Detection system initialized and ready',
            state: this.state
          };
        } else {
          throw new Error('Model loading failed');
        }
      } else {
        throw new Error(`Unexpected initialization status: ${response.data.status}`);
      }
    } catch (error) {
      console.error('❌ Error initializing detection processor:', error);
      this.resetToInitializing('Initialization failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Initialization timed out. Please check if the detection service is running.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to detection service. Please ensure the backend is running.');
      } else {
        throw new Error(`Failed to initialize detection processor: ${error.response?.data?.detail || error.message}`);
      }
    } finally {
      this.initializationPromise = null;
    }
  }

  // Updated loadModel method with initial check parameter
  async loadModel(isInitialCheck = false) {
    try {
      if (this.shouldSkipHealthCheck(isInitialCheck, false)) {
        return {
          success: false,
          message: 'Health check skipped due to system state'
        };
      }

      this.healthCheckInProgress = true;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.HEALTH_CHECK_TIMEOUT);

      const response = await fetch(`/api/detection/redis/health`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      this.lastHealthCheck = Date.now();
      
      // Mark initial health check as performed
      if (isInitialCheck) {
        this.hasPerformedInitialHealthCheck = true;
        console.log('✅ Initial health check completed and marked');
      }
      
      if (!response.ok) {
        if (response.status === 503) {
          console.log('🔄 Health check failed, model needs reloading...');
          throw new Error('Detection service not ready');
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }

      const result = await response.json();
      const modelLoaded = result.status === 'healthy';
      
      return {
        success: modelLoaded,
        message: modelLoaded ? 'Detection model loaded successfully' : 'Detection model not ready'
      };
      
    } catch (error) {
      console.error('Error loading detection model:', error);
      
      if (error.name === 'AbortError') {
        throw new Error('Health check timed out. Please check if the detection service is responding.');
      }
      
      throw new Error(`Failed to load detection model: ${error.message}`);
    } finally {
      this.healthCheckInProgress = false;
    }
  }

  async ensureCameraStarted(cameraId) {
    try {
      const numericCameraId = parseInt(cameraId, 10);
      if (isNaN(numericCameraId)) {
        throw new Error(`Invalid camera ID: ${cameraId}`);
      }

      console.log(`📹 Starting camera ${numericCameraId} for detection...`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.CAMERA_START_TIMEOUT);

      try {
        const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
          camera_id: numericCameraId
        }, {
          signal: controller.signal,
          timeout: this.CAMERA_START_TIMEOUT
        });
        
        clearTimeout(timeoutId);
        console.log("✅ Camera started for detection:", cameraResponse.data.message);

        await new Promise(resolve => setTimeout(resolve, 2000));

        return {
          success: true,
          message: cameraResponse.data.message
        };

      } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
          throw new Error(`Camera startup timed out after ${this.CAMERA_START_TIMEOUT / 1000} seconds`);
        }
        
        if (error.response?.status === 409 || error.response?.data?.detail?.includes('already running')) {
          console.log(`📹 Camera ${numericCameraId} is already running`);
          return {
            success: true,
            message: 'Camera already running'
          };
        }
        
        throw error;
      }

    } catch (error) {
      console.error(`❌ Error starting camera ${cameraId}:`, error);
      throw new Error(`Failed to start camera ${cameraId}: ${error.message}`);
    }
  }

  // Updated checkOptimizedHealth with proper parameters and flags
  async checkOptimizedHealth(isInitialCheck = false, isPostShutdownCheck = false) {
    if (this.shouldSkipHealthCheck(isInitialCheck, isPostShutdownCheck)) {
      return {
        streaming: { status: 'skipped', message: 'Health check skipped' },
        detection: { status: 'skipped', message: 'Health check skipped' },
        overall: false
      };
    }

    this.healthCheckInProgress = true;

    try {
      // Allow health checks during initialization and post-shutdown
      if (!this.isOperational() && !isInitialCheck && !isPostShutdownCheck) {
        throw new Error(`Cannot perform health check in state: ${this.state}`);
      }

      const healthCheckPromises = [
        Promise.race([
          api.get('/api/video_streaming/video/optimized/health'),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Streaming health check timeout')), 5000)
          )
        ]).catch(error => ({
          data: { status: 'unhealthy', error: error.message }
        })),
        
        Promise.race([
          api.get('/api/detection/redis/health'),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Detection health check timeout')), 5000)
          )
        ]).catch(error => ({
          data: { status: 'unhealthy', error: error.message }
        }))
      ];

      const [streamingHealth, detectionHealth] = await Promise.all(healthCheckPromises);
      this.lastHealthCheck = Date.now();

      // Mark post-shutdown health check as performed
      if (isPostShutdownCheck) {
        this.hasPerformedPostShutdownCheck = true;
        console.log('✅ Post-shutdown health check completed and marked');
      }

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const detectionHealthy = detectionHealth.data.status === 'healthy';

      console.log(`🩺 Health check completed - Streaming: ${streamingHealthy ? 'Healthy' : 'Unhealthy'}, Detection: ${detectionHealthy ? 'Healthy' : 'Unhealthy'}`);

      return {
        streaming: streamingHealth.data,
        detection: detectionHealth.data,
        overall: streamingHealthy && detectionHealthy
      };
    } catch (error) {
      console.error("Error checking optimized service health:", error);
      this.lastHealthCheck = Date.now();
      return {
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      };
    } finally {
      this.healthCheckInProgress = false;
    }
  }

  async gracefulShutdown() {
    try {
      console.log('🛑 Initiating graceful detection shutdown...');
      this.setState(DetectionStates.SHUTTING_DOWN, 'Graceful shutdown requested');
      
      // If not operational, already shut down
      if (!this.isOperational() && this.state !== DetectionStates.SHUTTING_DOWN) {
        console.log('ℹ️ Detection service already shut down');
        this.setState(DetectionStates.READY, 'Already shut down');
        return {
          success: true,
          message: 'Detection service was already shut down'
        };
      }
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.SHUTDOWN_TIMEOUT);

      const response = await api.post('/api/detection/detection/shutdown/graceful', {}, {
        signal: controller.signal,
        timeout: this.SHUTDOWN_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      // After successful shutdown, system is ready for new initialization
      this.isModelLoaded = false;
      this.setState(DetectionStates.READY, 'Shutdown completed');
      
      console.log('✅ Graceful detection shutdown completed:', response.data);
      
      return {
        success: true,
        message: 'Detection service shutdown completed',
        details: response.data
      };
      
    } catch (error) {
      console.error('❌ Error during graceful shutdown:', error);
      // On shutdown error, reset to initializing so we can try again
      this.resetToInitializing('Shutdown failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Graceful shutdown timed out. Detection service may still be running.');
      }
      
      throw new Error(`Graceful shutdown failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getShutdownStatus() {
    try {
      const response = await api.get('/api/detection/detection/shutdown/status');
      return response.data;
    } catch (error) {
      console.error('Error getting shutdown status:', error);
      throw new Error(`Failed to get shutdown status: ${error.response?.data?.detail || error.message}`);
    }
  }

  isReady() {
    return this.state === DetectionStates.READY && this.isModelLoaded;
  }

  isRunning() {
    return this.state === DetectionStates.RUNNING;
  }

  isInitializing() {
    return this.state === DetectionStates.INITIALIZING;
  }

  isShuttingDown() {
    return this.state === DetectionStates.SHUTTING_DOWN;
  }

  getDetailedStatus() {
    return {
      state: this.state,
      isModelLoaded: this.isModelLoaded,
      isReady: this.isReady(),
      isRunning: this.isRunning(),
      isInitializing: this.isInitializing(),
      isShuttingDown: this.isShuttingDown(),
      activeStreams: this.currentStreams.size,
      lastHealthCheck: this.lastHealthCheck,
      healthCheckInProgress: this.healthCheckInProgress,
      hasPerformedInitialHealthCheck: this.hasPerformedInitialHealthCheck,
      hasPerformedPostShutdownCheck: this.hasPerformedPostShutdownCheck
    };
  }

  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    const {
      detectionFps = 5.0,
      streamQuality = 85,
      priority = 1
    } = options;

    try {
      console.log(`🎯 Starting optimized detection feed for camera ${cameraId} with target: ${targetLabel}`);

      // Ensure system is ready
      if (!this.canStart()) {
        throw new Error(`Cannot start detection in state: ${this.state}. Current state must be READY.`);
      }

      if (!this.isModelLoaded) {
        throw new Error('Detection model is not loaded');
      }

      // Start camera
      try {
        await this.ensureCameraStarted(cameraId);
      } catch (cameraError) {
        throw new Error(`Camera startup failed: ${cameraError.message}`);
      }

      // Stop any existing stream for this camera
      await this.stopOptimizedDetectionFeed(cameraId, false);

      const params = new URLSearchParams({
        target_label: targetLabel,
        detection_fps: detectionFps.toString(),
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream_with_detection/${cameraId}?${params}`;
      
      const streamKey = `${cameraId}_${targetLabel}`;
      this.detectionStats.set(streamKey, {
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: streamQuality,
        detectionFps: detectionFps
      });

      this.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel,
        streamKey,
        startTime: Date.now(),
        isActive: true
      });

      // Update state to running
      this.setState(DetectionStates.RUNNING, `Started detection for camera ${cameraId}`);

      this.startStatsMonitoring(cameraId, streamKey);

      console.log(`✅ Successfully started optimized detection feed for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting optimized detection feed:", error);
      
      try {
        await this.stopOptimizedDetectionFeed(cameraId, false);
      } catch (cleanupError) {
        console.error("Error during cleanup:", cleanupError);
      }
      
      throw new Error(`Failed to start detection feed: ${error.message}`);
    }
  }

  async startOptimizedStream(cameraId, options = {}) {
    const { streamQuality = 85 } = options;

    try {
      console.log(`📺 Starting optimized stream for camera ${cameraId}`);

      if (!this.canStart()) {
        throw new Error(`Cannot start stream in state: ${this.state}. Current state must be READY.`);
      }

      try {
        await this.ensureCameraStarted(cameraId);
      } catch (cameraError) {
        throw new Error(`Camera startup failed: ${cameraError.message}`);
      }

      await this.stopOptimizedStream(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream/${cameraId}?${params}`;
      
      this.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true
      });

      // Update state to running if not already
      if (this.state === DetectionStates.READY) {
        this.setState(DetectionStates.RUNNING, `Started stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started optimized stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting optimized stream:", error);
      throw new Error(`Failed to start stream: ${error.message}`);
    }
  }

  async stopOptimizedDetectionFeed(cameraId, performShutdown = false) {
    try {
      const stream = this.currentStreams.get(cameraId);
      if (stream) {
        console.log(`⏹️ Stopping optimized detection feed for camera ${cameraId}`);
        
        this.stopStatsMonitoring(cameraId);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping stream API for camera ${cameraId}:`, error.message);
        }
        
        this.currentStreams.delete(cameraId);
        if (stream.streamKey) {
          this.detectionStats.delete(stream.streamKey);
        }
        
        // Update state based on remaining streams
        if (this.currentStreams.size === 0 && this.state === DetectionStates.RUNNING) {
          this.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        if (performShutdown) {
          try {
            console.log('🔄 Performing graceful detection shutdown...');
            await this.gracefulShutdown();
            console.log('✅ Detection service gracefully shut down');
          } catch (shutdownError) {
            console.error('⚠️ Graceful shutdown failed, but stream stopped:', shutdownError.message);
          }
        }
        
        console.log(`✅ Successfully stopped detection feed for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized detection feed:", error);
      throw error;
    }
  }

  async stopOptimizedStream(cameraId) {
    try {
      const stream = this.currentStreams.get(cameraId);
      if (stream) {
        console.log(`⏹️ Stopping optimized stream for camera ${cameraId}`);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping stream API for camera ${cameraId}:`, error.message);
        }
        
        this.currentStreams.delete(cameraId);
        
        // Update state based on remaining streams
        if (this.currentStreams.size === 0 && this.state === DetectionStates.RUNNING) {
          this.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        console.log(`✅ Successfully stopped stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized stream:", error);
      throw error;
    }
  }

  async stopAllStreams(performShutdown = true) {
    try {
      console.log('🛑 Stopping all optimized streams...');
      
      const stopPromises = Array.from(this.currentStreams.keys()).map(cameraId => 
        this.stopOptimizedDetectionFeed(cameraId, false)
      );
      
      await Promise.allSettled(stopPromises);
      
      try {
        await api.post('/api/video_streaming/video/optimized/streams/stop_all');
      } catch (error) {
        console.warn('⚠️ Error calling stop_all API:', error.message);
      }
      
      this.currentStreams.clear();
      this.detectionStats.clear();
      for (const cameraId of this.eventListeners.keys()) {
        this.stopStatsMonitoring(cameraId);
      }
      
      // Update state
      if (this.state === DetectionStates.RUNNING) {
        this.setState(DetectionStates.READY, 'All streams stopped');
      }
      
      if (performShutdown) {
        try {
          console.log('🔄 Performing graceful detection shutdown after stopping all streams...');
          await this.gracefulShutdown();
          console.log('✅ All streams stopped and detection service gracefully shut down');
        } catch (shutdownError) {
          console.error('⚠️ Graceful shutdown failed, but all streams stopped:', shutdownError.message);
        }
      }
      
      console.log("✅ Stopped all optimized streams");
    } catch (error) {
      console.error("❌ Error stopping all streams:", error);
      throw error;
    }
  }

  async reloadModel() {
    try {
      const response = await api.post("/api/detection/detection/model/reload");
      console.log("✅ Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("❌ Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  }

  // Stats monitoring methods (unchanged)
  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
      if (this.state === DetectionStates.SHUTTING_DOWN) {
        console.log(`⏭️ Skipping stats update for camera ${cameraId} - system shutting down`);
        return;
      }

      try {
        const response = await api.get(`/api/video_streaming/video/optimized/stats/${cameraId}`);
        const streamStats = response.data.streams?.[0];
        
        if (streamStats && this.currentStreams.has(cameraId)) {
          const currentStats = this.detectionStats.get(streamKey) || {};
          
          const updatedStats = {
            ...currentStats,
            objectDetected: this.determineDetectionStatus(streamStats),
            detectionCount: streamStats.detections_processed || currentStats.detectionCount,
            nonTargetCount: streamStats.non_target_count || currentStats.nonTargetCount,
            lastDetectionTime: streamStats.last_detection,
            avgProcessingTime: streamStats.avg_detection_time_ms || 0,
            streamQuality: streamStats.current_quality || currentStats.streamQuality,
            detectionFps: streamStats.detection_interval ? (25 / streamStats.detection_interval) : currentStats.detectionFps,
            queueDepth: streamStats.detection_backlog || 0,
            isStreamActive: streamStats.is_active
          };

          this.detectionStats.set(streamKey, updatedStats);
          this.notifyStatsListeners(cameraId, updatedStats);
        }
      } catch (error) {
        if (this.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error polling detection stats:", error);
        }
      }
    }, 2000);

    if (!this.eventListeners.has(cameraId)) {
      this.eventListeners.set(cameraId, { pollInterval, listeners: [] });
    } else {
      this.eventListeners.get(cameraId).pollInterval = pollInterval;
    }
  };

  stopStatsMonitoring = (cameraId) => {
    const eventData = this.eventListeners.get(cameraId);
    if (eventData?.pollInterval) {
      clearInterval(eventData.pollInterval);
    }
    this.eventListeners.delete(cameraId);
  };

  addStatsListener = (cameraId, callback) => {
    if (!this.eventListeners.has(cameraId)) {
      this.eventListeners.set(cameraId, { listeners: [] });
    }
    this.eventListeners.get(cameraId).listeners.push(callback);
  };

  removeStatsListener = (cameraId, callback) => {
    const eventData = this.eventListeners.get(cameraId);
    if (eventData) {
      eventData.listeners = eventData.listeners.filter(cb => cb !== callback);
    }
  };

  notifyStatsListeners = (cameraId, stats) => {
    const eventData = this.eventListeners.get(cameraId);
    if (eventData?.listeners) {
      eventData.listeners.forEach(callback => {
        try {
          callback(stats);
        } catch (error) {
          console.error("Error in stats listener callback:", error);
        }
      });
    }
  };

  determineDetectionStatus = (streamStats) => {
    if (!streamStats.last_detection) return false;
    const lastDetectionTime = new Date(streamStats.last_detection).getTime();
    const now = Date.now();
    return (now - lastDetectionTime) < 5000;
  };

  getDetectionStats = (cameraId) => {
    const stream = this.currentStreams.get(cameraId);
    if (stream?.streamKey) {
      return this.detectionStats.get(stream.streamKey) || {
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0
      };
    }
    return null;
  };

  async getAllStreamingStats() {
    try {
      if (this.state === DetectionStates.SHUTTING_DOWN) {
        return {
          active_streams: 0,
          avg_processing_time_ms: 0,
          total_detections: 0,
          system_load_percent: 0,
          memory_usage_mb: 0
        };
      }
      const response = await api.get('/api/video_streaming/video/optimized/stats');
      return response.data;
    } catch (error) { 
      console.error("❌ Error getting streaming stats:", error);
      throw error;
    }
  }

  getPerformanceComparison = async () => {
    try {
      const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
      return response.data;
    } catch (error) {
      console.error("❌ Error getting performance comparison:", error);
      throw error;
    }
  };

  // Legacy method compatibility
  startDetectionFeed = async (cameraId, targetLabel) => {
    return this.startOptimizedDetectionFeed(cameraId, targetLabel);
  };

  stopDetectionFeed = async (cameraId, performShutdown = true) => {
    return this.stopOptimizedDetectionFeed(cameraId, performShutdown);
  };

  stopVideoStream = async (cameraId, performShutdown = true) => {
    return this.stopOptimizedDetectionFeed(cameraId, performShutdown);
  };

  cleanup = async () => { 
    try {
      console.log('🧹 Starting DetectionService cleanup...');
      await this.stopAllStreams(true);
      console.log('✅ DetectionService cleanup completed');
    } catch (error) {
      console.error("❌ Error during cleanup:", error);
    }
  };
}

export const detectionService = new DetectionService();