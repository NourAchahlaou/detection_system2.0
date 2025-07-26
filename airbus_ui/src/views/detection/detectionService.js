// detectionService.js - Fixed service with camera startup integration
import api from "../../utils/UseAxios";

class DetectionService {
  constructor() {
    this.currentStreams = new Map();
    this.detectionStats = new Map();
    this.wsConnections = new Map();
    this.eventListeners = new Map();
    
    // Fixed initialization state management
    this.initializationPromise = null;
    this.isInitializing = false;
    this.isInitialized = false;
    this.isModelLoaded = false;
    
    // Add timeout handling
    this.INITIALIZATION_TIMEOUT = 30000; // 30 seconds
    this.HEALTH_CHECK_TIMEOUT = 10000; // 10 seconds
    this.CAMERA_START_TIMEOUT = 15000; // 15 seconds for camera startup
  }

  // Auto-initialize detection processor with proper error handling
  async ensureInitialized() {
    // If already initialized, return immediately
    if (this.isInitialized && this.isModelLoaded) {
      return { success: true, message: 'Already initialized' };
    }

    // If initialization is in progress, wait for it with timeout
    if (this.initializationPromise) {
      try {
        return await Promise.race([
          this.initializationPromise,
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Initialization timeout')), this.INITIALIZATION_TIMEOUT)
          )
        ]);
      } catch (error) {
        // Reset on timeout/error so retry is possible
        this.initializationPromise = null;
        this.isInitializing = false;
        throw error;
      }
    }

    // Start initialization
    this.initializationPromise = this.initializeProcessor();
    return await this.initializationPromise;
  }

  // Initialize the detection processor with timeout
  async initializeProcessor() {
    if (this.isInitializing) {
      throw new Error('Initialization already in progress');
    }

    this.isInitializing = true;
    console.log('Initializing detection processor...');

    try {
      // Add timeout to the initialization request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.INITIALIZATION_TIMEOUT);

      const response = await api.post('/api/detection/redis/initialize', {}, {
        signal: controller.signal,
        timeout: this.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running') {
        this.isInitialized = true;
        this.isModelLoaded = true; // Set model as loaded when processor is initialized
        console.log('Detection processor initialized successfully:', response.data.message);
        return {
          success: true,
          message: response.data.message,
          status: response.data.status
        };
      } else {
        throw new Error(`Unexpected initialization status: ${response.data.status}`);
      }
    } catch (error) {
      console.error('Error initializing detection processor:', error);
      this.isInitialized = false;
      this.isModelLoaded = false;
      
      // Reset promises so retry is possible
      this.initializationPromise = null;
      
      // Handle different error types
      if (error.name === 'AbortError') {
        throw new Error('Initialization timed out. Please check if the detection service is running.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to detection service. Please ensure the backend is running.');
      } else {
        throw new Error(`Failed to initialize detection processor: ${error.response?.data?.detail || error.message}`);
      }
    } finally {
      this.isInitializing = false;
    }
  }

  // Enhanced model loading with better error handling
  async loadModel() {
    try {
      // First, ensure the processor is initialized
      const initResult = await this.ensureInitialized();
      if (!initResult.success) {
        throw new Error(initResult.message || 'Failed to initialize processor');
      }

      // Then check health with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.HEALTH_CHECK_TIMEOUT);

      const response = await fetch(`/api/detection/redis/health`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        // If health check fails, try to initialize again
        if (response.status === 503) {
          console.log('Health check failed, attempting re-initialization...');
          
          // Reset initialization state for retry
          this.isInitialized = false;
          this.isModelLoaded = false;
          this.initializationPromise = null;
          
          await this.initializeProcessor();
          
          // Retry health check
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
      
      // Handle timeout errors specifically
      if (error.name === 'AbortError') {
        throw new Error('Health check timed out. Please check if the detection service is responding.');
      }
      
      throw new Error(`Failed to load detection model: ${error.message}`);
    }
  }

  // Ensure camera is started and ready
  async ensureCameraStarted(cameraId) {
    try {
      const numericCameraId = parseInt(cameraId, 10);
      if (isNaN(numericCameraId)) {
        throw new Error(`Invalid camera ID: ${cameraId}`);
      }

      console.log(`Starting camera ${numericCameraId} for detection...`);

      // Start camera via artifact keeper (handles hardware initialization)
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
        console.log("Camera started for detection:", cameraResponse.data.message);

        // Wait a moment for camera to fully initialize
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
        
        // If camera is already running, that's okay
        if (error.response?.status === 409 || error.response?.data?.detail?.includes('already running')) {
          console.log(`Camera ${numericCameraId} is already running`);
          return {
            success: true,
            message: 'Camera already running'
          };
        }
        
        throw error;
      }

    } catch (error) {
      console.error(`Error starting camera ${cameraId}:`, error);
      throw new Error(`Failed to start camera ${cameraId}: ${error.message}`);
    }
  }

  // Enhanced health check with proper timeout handling
  async checkOptimizedHealth() {
    try {
      // Ensure processor is initialized before health checks
      await this.ensureInitialized();

      const healthCheckPromises = [
        // Streaming health check
        Promise.race([
          api.get('/api/video_streaming/video/optimized/health'),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Streaming health check timeout')), 5000)
          )
        ]).catch(error => ({
          data: { status: 'unhealthy', error: error.message }
        })),
        
        // Detection health check
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

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const detectionHealthy = detectionHealth.data.status === 'healthy';

      return {
        streaming: streamingHealth.data,
        detection: detectionHealth.data,
        overall: streamingHealthy && detectionHealthy
      };
    } catch (error) {
      console.error("Error checking optimized service health:", error);
      return {
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false
      };
    }
  }

  // Force reset initialization state (for manual retry)
  resetInitialization() {
    this.isInitialized = false;
    this.isModelLoaded = false;
    this.isInitializing = false;
    this.initializationPromise = null;
    console.log('Detection service initialization state reset');
  }

  // Check if service is ready for operations
  isReady() {
    return this.isInitialized && this.isModelLoaded && !this.isInitializing;
  }

  // Get current initialization status
  getInitializationStatus() {
    return {
      isInitialized: this.isInitialized,
      isModelLoaded: this.isModelLoaded,
      isInitializing: this.isInitializing,
      isReady: this.isReady()
    };
  }

  // Shutdown processor
  async shutdownProcessor() {
    try {
      const response = await api.post('/api/detection/redis/shutdown');
      this.isInitialized = false;
      this.isModelLoaded = false;
      this.initializationPromise = null;
      console.log('Detection processor shutdown:', response.data.message);
      return response.data;
    } catch (error) {
      console.error('Error shutting down detection processor:', error);
      throw error;
    }
  }

  // Get processor status
  async getProcessorStatus() {
    try {
      const response = await api.get('/api/detection/redis/status');
      return response.data;
    } catch (error) {
      console.error('Error getting processor status:', error);
      throw error;
    }
  }

  // Enhanced detection feed start with camera startup integration
  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    const {
      detectionFps = 5.0,
      streamQuality = 85,
      priority = 1
    } = options;

    try {
      console.log(`Starting optimized detection feed for camera ${cameraId} with target: ${targetLabel}`);

      // Step 1: Ensure processor is initialized
      const initResult = await this.ensureInitialized();
      if (!initResult.success) {
        throw new Error(`Cannot start detection: ${initResult.message}`);
      }

      // Step 2: Verify service is ready
      if (!this.isReady()) {
        throw new Error('Detection service is not ready. Please wait for initialization to complete.');
      }

      // Step 3: Start camera hardware (this was missing!)
      try {
        await this.ensureCameraStarted(cameraId);
      } catch (cameraError) {
        throw new Error(`Camera startup failed: ${cameraError.message}`);
      }

      // Step 4: Stop any existing stream for this camera first
      await this.stopOptimizedDetectionFeed(cameraId);

      // Step 5: Start the optimized detection stream
      const params = new URLSearchParams({
        target_label: targetLabel,
        detection_fps: detectionFps.toString(),
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream_with_detection/${cameraId}?${params}`;
      
      // Step 6: Initialize stats tracking
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

      // Step 7: Start real-time stats monitoring
      this.startStatsMonitoring(cameraId, streamKey);

      console.log(`Successfully started optimized detection feed for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("Error starting optimized detection feed:", error);
      
      // Cleanup on failure
      try {
        await this.stopOptimizedDetectionFeed(cameraId);
      } catch (cleanupError) {
        console.error("Error during cleanup:", cleanupError);
      }
      
      throw new Error(`Failed to start detection feed: ${error.message}`);
    }
  }

  // Enhanced stream start with camera startup
  startOptimizedStream = async (cameraId, options = {}) => {
    const { streamQuality = 85 } = options;

    try {
      console.log(`Starting optimized stream for camera ${cameraId}`);

      // Step 1: Start camera hardware
      try {
        await this.ensureCameraStarted(cameraId);
      } catch (cameraError) {
        throw new Error(`Camera startup failed: ${cameraError.message}`);
      }

      // Step 2: Stop existing stream
      await this.stopOptimizedStream(cameraId);

      // Step 3: Start new stream
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

      console.log(`Successfully started optimized stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("Error starting optimized stream:", error);
      throw new Error(`Failed to start stream: ${error.message}`);
    }
  };

  // Enhanced stop methods with camera cleanup
  stopOptimizedDetectionFeed = async (cameraId) => {
    try {
      const stream = this.currentStreams.get(cameraId);
      if (stream) {
        console.log(`Stopping optimized detection feed for camera ${cameraId}`);
        
        // Stop stats monitoring
        this.stopStatsMonitoring(cameraId);
        
        // Stop stream
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`Error stopping stream API for camera ${cameraId}:`, error.message);
        }
        
        // Clean up local state
        this.currentStreams.delete(cameraId);
        if (stream.streamKey) {
          this.detectionStats.delete(stream.streamKey);
        }
        
        console.log(`Successfully stopped detection feed for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("Error stopping optimized detection feed:", error);
      throw error;
    }
  };

  stopOptimizedStream = async (cameraId) => {
    try {
      const stream = this.currentStreams.get(cameraId);
      if (stream) {
        console.log(`Stopping optimized stream for camera ${cameraId}`);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`Error stopping stream API for camera ${cameraId}:`, error.message);
        }
        
        this.currentStreams.delete(cameraId);
        console.log(`Successfully stopped stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("Error stopping optimized stream:", error);
      throw error;
    }
  };

  // Rest of the methods remain the same...
  reloadModel = async () => {
    try {
      const response = await api.post("/api/detection/detection/model/reload");
      console.log("Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  };

  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
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
        console.debug("Error polling detection stats:", error);
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

  getAllStreamingStats = async () => {
    try {
      const response = await api.get('/api/video_streaming/video/optimized/stats');
      return response.data;
    } catch (error) {
      console.error("Error getting streaming stats:", error);
      throw error;
    }
  };

  getPerformanceComparison = async () => {
    try {
      const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
      return response.data;
    } catch (error) {
      console.error("Error getting performance comparison:", error);
      throw error;
    }
  };

  stopAllStreams = async () => {
    try {
      await api.post('/api/video_streaming/video/optimized/streams/stop_all');
      this.currentStreams.clear();
      this.detectionStats.clear();
      for (const cameraId of this.eventListeners.keys()) {
        this.stopStatsMonitoring(cameraId);
      }
      console.log("Stopped all optimized streams");
    } catch (error) {
      console.error("Error stopping all streams:", error);
      throw error;
    }
  };

  // Legacy methods for backward compatibility
  startDetectionFeed = async (cameraId, targetLabel) => {
    return this.startOptimizedDetectionFeed(cameraId, targetLabel);
  };

  stopDetectionFeed = async (cameraId) => {
    return this.stopOptimizedDetectionFeed(cameraId);
  };

  stopVideoStream = async (cameraId) => {
    return this.stopOptimizedDetectionFeed(cameraId);
  };

  cleanup = async () => {
    try {
      await this.stopAllStreams();
      // Optionally shutdown processor on cleanup
      // await this.shutdownProcessor();
    } catch (error) {
      console.error("Error during cleanup:", error);
    }
  };
}

// Export singleton instance
export const detectionService = new DetectionService();