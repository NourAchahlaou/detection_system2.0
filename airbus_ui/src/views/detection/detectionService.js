import api from "../../utils/UseAxios";

class DetectionService {
  constructor() {
    this.currentStreams = new Map();
    this.detectionStats = new Map();
    this.wsConnections = new Map();
    this.eventListeners = new Map();
    
    this.initializationPromise = null;
    this.isInitializing = false;
    this.isInitialized = false;
    this.isModelLoaded = false;
    this.isShuttingDown = false;
    this.lastHealthCheck = null;
    this.healthCheckInProgress = false;
    
    this.INITIALIZATION_TIMEOUT = 30000;
    this.HEALTH_CHECK_TIMEOUT = 10000;
    this.CAMERA_START_TIMEOUT = 15000;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.HEALTH_CHECK_COOLDOWN = 5000;
  }

  resetAllState() {
    this.isInitialized = false;
    this.isModelLoaded = false;
    this.isInitializing = false;
    this.isShuttingDown = false;
    this.initializationPromise = null;
    this.lastHealthCheck = null;
    this.healthCheckInProgress = false;
    console.log('🔄 Detection service state completely reset');
  }

  setShuttingDown(isShuttingDown) {
    this.isShuttingDown = isShuttingDown;
    console.log(`🔄 Shutdown state set to: ${isShuttingDown}`);
  }

  shouldSkipHealthCheck() {
    if (this.isShuttingDown) {
      console.log('⏭️ Skipping health check - system is shutting down');
      return true;
    }

    if (this.healthCheckInProgress) {
      console.log('⏭️ Skipping health check - already in progress');
      return true;
    }

    if (this.lastHealthCheck && (Date.now() - this.lastHealthCheck) < this.HEALTH_CHECK_COOLDOWN) {
      console.log('⏭️ Skipping health check - too soon since last check');
      return true;
    }

    return false;
  }

  async ensureInitialized() {
    if (this.isInitialized && this.isModelLoaded) {
      return { success: true, message: 'Already initialized' };
    }

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
        this.resetAllState();
        throw error;
      }
    }

    this.initializationPromise = this.initializeProcessor();
    return await this.initializationPromise;
  }

  async initializeProcessor() {
    if (this.isInitializing) {
      throw new Error('Initialization already in progress');
    }

    this.isInitializing = true;
    this.isShuttingDown = false;
    console.log('🚀 Initializing detection processor...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.INITIALIZATION_TIMEOUT);

      const response = await api.post('/api/detection/redis/initialize', {}, {
        signal: controller.signal,
        timeout: this.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running') {
        this.isInitialized = true;
        this.isModelLoaded = true;
        console.log('✅ Detection processor initialized successfully:', response.data.message);
        return {
          success: true,
          message: response.data.message,
          status: response.data.status
        };
      } else {
        throw new Error(`Unexpected initialization status: ${response.data.status}`);
      }
    } catch (error) {
      console.error('❌ Error initializing detection processor:', error);
      this.resetAllState();
      
      if (error.name === 'AbortError') {
        throw new Error('Initialization timed out. Please check if the detection service is running.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to detection service. Please ensure the backend is running.');
      } else {
        throw new Error(`Failed to initialize detection processor: ${error.response?.data?.detail || error.message}`);
      }
    } finally {
      this.isInitializing = false;
      if (this.isInitialized) {
        this.initializationPromise = null;
      }
    }
  }

  async loadModel() {
    try {
      const initResult = await this.ensureInitialized();
      if (!initResult.success) {
        throw new Error(initResult.message || 'Failed to initialize processor');
      }

      if (this.shouldSkipHealthCheck()) {
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
      
      if (!response.ok) {
        if (response.status === 503) {
          console.log('🔄 Health check failed, attempting re-initialization...');
          this.resetAllState();
          await this.initializeProcessor();
          
          const retryController = new AbortController();
          const retryTimeoutId = setTimeout(() => retryController.abort(), this.HEALTH_CHECK_TIMEOUT);
          
          const retryResponse = await fetch(`/api/detection/redis/health`, {
            signal: retryController.signal
          });
          
          clearTimeout(retryTimeoutId);
          
          if (!retryResponse.ok) {
            throw new Error(`Health check failed after initialization: ${retryResponse.status}`);
          }
          const retryResult = await retryResponse.json();
          this.isModelLoaded = retryResult.status === 'healthy';
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      } else {
        const result = await response.json();
        this.isModelLoaded = result.status === 'healthy';
      }
      
      return {
        success: this.isModelLoaded,
        message: this.isModelLoaded ? 'Detection model loaded successfully' : 'Detection model not ready'
      };
      
    } catch (error) {
      console.error('Error loading detection model:', error);
      this.isModelLoaded = false;
      
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

  async checkOptimizedHealth() {
    if (this.shouldSkipHealthCheck()) {
      return {
        streaming: { status: 'skipped', message: 'Health check skipped' },
        detection: { status: 'skipped', message: 'Health check skipped' },
        overall: false
      };
    }

    this.healthCheckInProgress = true;

    try {
      await this.ensureInitialized();

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

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const detectionHealthy = detectionHealth.data.status === 'healthy';

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
      this.setShuttingDown(true);
      
      if (!this.isInitialized && !this.isInitializing) {
        console.log('ℹ️ Detection service already shut down');
        this.resetAllState();
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
      this.resetAllState();
      
      console.log('✅ Graceful detection shutdown completed:', response.data);
      
      return {
        success: true,
        message: 'Detection service shutdown completed',
        details: response.data
      };
      
    } catch (error) {
      console.error('❌ Error during graceful shutdown:', error);
      this.resetAllState();
      
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

  resetInitialization() {
    this.resetAllState();
    console.log('🔄 Detection service initialization state manually reset');
  }

  isReady() {
    return this.isInitialized && this.isModelLoaded && !this.isInitializing && !this.isShuttingDown;
  }

  getInitializationStatus() {
    return {
      isInitialized: this.isInitialized,
      isModelLoaded: this.isModelLoaded,
      isInitializing: this.isInitializing,
      isShuttingDown: this.isShuttingDown,
      isReady: this.isReady(),
      hasInitializationPromise: !!this.initializationPromise,
      lastHealthCheck: this.lastHealthCheck
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

      const initResult = await this.ensureInitialized();
      if (!initResult.success) {
        throw new Error(`Cannot start detection: ${initResult.message}`);
      }

      if (!this.isReady()) {
        throw new Error('Detection service is not ready. Please wait for initialization to complete.');
      }

      try {
        await this.ensureCameraStarted(cameraId);
      } catch (cameraError) {
        throw new Error(`Camera startup failed: ${cameraError.message}`);
      }

      await this.stopOptimizedDetectionFeed(cameraId);

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

      this.startStatsMonitoring(cameraId, streamKey);

      console.log(`✅ Successfully started optimized detection feed for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting optimized detection feed:", error);
      
      try {
        await this.stopOptimizedDetectionFeed(cameraId);
      } catch (cleanupError) {
        console.error("Error during cleanup:", cleanupError);
      }
      
      throw new Error(`Failed to start detection feed: ${error.message}`);
    }
  }

  startOptimizedStream = async (cameraId, options = {}) => {
    const { streamQuality = 85 } = options;

    try {
      console.log(`📺 Starting optimized stream for camera ${cameraId}`);

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

      console.log(`✅ Successfully started optimized stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting optimized stream:", error);
      throw new Error(`Failed to start stream: ${error.message}`);
    }
  };

  stopOptimizedDetectionFeed = async (cameraId, performShutdown = false) => {
    try {
      this.setShuttingDown(true);
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
        
        if (performShutdown || this.currentStreams.size === 0) {
          try {
            console.log('🔄 Performing graceful detection shutdown...');
            await this.gracefulShutdown();
            console.log('✅ Detection service gracefully shut down');
          } catch (shutdownError) {
            console.error('⚠️ Graceful shutdown failed, but stream stopped:', shutdownError.message);
          }
        } else {
          this.setShuttingDown(false);
        }
        
        console.log(`✅ Successfully stopped detection feed for camera ${cameraId}`);
      } else {
        this.setShuttingDown(false);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized detection feed:", error);
      this.setShuttingDown(false);
      throw error;
    }
  };

  stopOptimizedStream = async (cameraId) => {
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
        console.log(`✅ Successfully stopped stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized stream:", error);
      throw error;
    }
  };

  stopAllStreams = async (performShutdown = true) => {
    try {
      console.log('🛑 Stopping all optimized streams...');
      this.setShuttingDown(true);
      
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
      
      if (performShutdown) {
        try {
          console.log('🔄 Performing graceful detection shutdown after stopping all streams...');
          await this.gracefulShutdown();
          console.log('✅ All streams stopped and detection service gracefully shut down');
        } catch (shutdownError) {
          console.error('⚠️ Graceful shutdown failed, but all streams stopped:', shutdownError.message);
        }
      } else {
        this.setShuttingDown(false);
      }
      
      console.log("✅ Stopped all optimized streams");
    } catch (error) {
      console.error("❌ Error stopping all streams:", error);
      this.setShuttingDown(false);
      throw error;
    }
  };

  reloadModel = async () => {
    try {
      const response = await api.post("/api/detection/detection/model/reload");
      console.log("✅ Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("❌ Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  };

  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
      if (this.isShuttingDown) {
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
        if (!this.isShuttingDown) {
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
      if (this.isShuttingDown) {
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
  };

  getPerformanceComparison = async () => {
    try {
      const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
      return response.data;
    } catch (error) {
      console.error("❌ Error getting performance comparison:", error);
      throw error;
    }
  };

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