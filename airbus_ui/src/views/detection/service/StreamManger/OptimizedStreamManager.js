import api from "../../../../utils/UseAxios";

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class OptimizedStreamManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  // ===================
  // OPTIMIZED STREAMING METHODS
  // ===================

  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    const {
      detectionFps = 5.0,
      streamQuality = 85,
      priority = 1
    } = options;

    try {
      console.log(`🎯 Starting optimized detection feed for camera ${cameraId} with target: ${targetLabel}`);

      // Enhanced state check
      if (this.detectionService.state !== DetectionStates.READY) {
        throw new Error(`Cannot start detection in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      // More robust readiness check
      console.log('🔧 Verifying detection service readiness...');
      const readinessCheck = await this.detectionService.ensureDetectionServiceReady();
      
      if (!readinessCheck.success) {
        if (readinessCheck.fallbackMode === 'basic') {
          throw new Error('Detection service not ready, basic mode recommended');
        } else {
          throw new Error(`Detection service not ready: ${readinessCheck.message}`);
        }
      }

      // Verify model is actually loaded
      if (!this.detectionService.isModelLoaded) {
        console.log('🔧 Model not loaded, attempting to load...');
        const modelResult = await this.detectionService.loadModel();
        if (!modelResult.success) {
          throw new Error(`Failed to load model: ${modelResult.message}`);
        }
      }

      // Double-check system supports optimized mode
      if (this.detectionService.shouldUseBasicMode()) {
        throw new Error('System in basic mode, cannot use optimized detection');
      }

      await this.ensureCameraStarted(cameraId);

      // Stop any existing optimized detection
      await this.stopOptimizedDetectionFeed(cameraId, false);

      // Add a small delay to ensure cleanup is complete
      await new Promise(resolve => setTimeout(resolve, 1000));

      const params = new URLSearchParams({
        target_label: targetLabel,
        detection_fps: detectionFps.toString(),
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream_with_detection/${cameraId}?${params}`;
      
      const streamKey = `${cameraId}_${targetLabel}`;
      this.detectionService.detectionStats.set(streamKey, {
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        streamQuality: streamQuality,
        detectionFps: detectionFps,
        mode: 'optimized'
      });

      this.detectionService.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel,
        streamKey,
        startTime: Date.now(),
        isActive: true,
        type: 'optimized_detection'
      });

      this.detectionService.setState(DetectionStates.RUNNING, `Started optimized detection for camera ${cameraId}`);
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
      
      throw new Error(`Failed to start optimized detection feed: ${error.message}`);
    }
  }

  async startOptimizedStream(cameraId, options = {}) {
    const { streamQuality = 85 } = options;

    try {
      console.log(`📺 Starting optimized stream for camera ${cameraId}`);

      if (!this.detectionService.canStart()) {
        throw new Error(`Cannot start stream in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      // Check if system supports optimized mode
      if (this.detectionService.shouldUseBasicMode()) {
        throw new Error('System in basic mode, cannot use optimized streaming');
      }

      await this.ensureCameraStarted(cameraId);
      await this.stopOptimizedStream(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream/${cameraId}?${params}`;
      
      this.detectionService.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `optimized_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'optimized_stream'
      });

      if (this.detectionService.state === DetectionStates.READY) {
        this.detectionService.setState(DetectionStates.RUNNING, `Started optimized stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started optimized stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting optimized stream:", error);
      throw new Error(`Failed to start optimized stream: ${error.message}`);
    }
  }

  async stopOptimizedDetectionFeed(cameraId, performShutdown = false) {
    try {
      const stream = this.detectionService.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.includes('optimized')) {
        console.log(`⏹️ Stopping optimized detection feed for camera ${cameraId}`);
        
        this.stopStatsMonitoring(cameraId);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping optimized stream API for camera ${cameraId}:`, error.message);
        }
        
        this.detectionService.currentStreams.delete(cameraId);
        if (stream.streamKey) {
          this.detectionService.detectionStats.delete(stream.streamKey);
        }
        
        if (this.detectionService.currentStreams.size === 0 && this.detectionService.state === DetectionStates.RUNNING) {
          this.detectionService.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        if (performShutdown && !this.detectionService.shouldUseBasicMode()) {
          try {
            console.log('🔄 Performing graceful detection shutdown...');
            await this.detectionService.gracefulShutdown();
            console.log('✅ Detection service gracefully shut down');
          } catch (shutdownError) {
            console.error('⚠️ Graceful shutdown failed, but stream stopped:', shutdownError.message);
          }
        }
        
        console.log(`✅ Successfully stopped optimized detection feed for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized detection feed:", error);
      throw error;
    }
  }

  async stopOptimizedStream(cameraId) {
    try {
      const stream = this.detectionService.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.includes('optimized')) {
        console.log(`⏹️ Stopping optimized stream for camera ${cameraId}`);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
          await api.post("/api/artifact_keeper/camera/stop");
        } catch (error) {
          console.warn(`⚠️ Error stopping optimized stream API for camera ${cameraId}:`, error.message);
        }
        
        this.detectionService.currentStreams.delete(cameraId);
        
        if (this.detectionService.currentStreams.size === 0 && this.detectionService.state === DetectionStates.RUNNING) {
          this.detectionService.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        console.log(`✅ Successfully stopped optimized stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping optimized stream:", error);
      throw error;
    }
  }

  async ensureCameraStarted(cameraId) {
    try {
      const numericCameraId = parseInt(cameraId, 10);
      if (isNaN(numericCameraId)) {
        throw new Error(`Invalid camera ID: ${cameraId}`);
      }

      console.log(`📹 Starting camera ${numericCameraId} for optimized detection...`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.CAMERA_START_TIMEOUT);

      try {
        const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
          camera_id: numericCameraId
        }, {
          signal: controller.signal,
          timeout: this.detectionService.CAMERA_START_TIMEOUT
        });
        
        clearTimeout(timeoutId);
        console.log("✅ Camera started for optimized detection:", cameraResponse.data.message);

        await new Promise(resolve => setTimeout(resolve, 2000));

        return {
          success: true,
          message: cameraResponse.data.message
        };

      } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
          throw new Error(`Camera startup timed out after ${this.detectionService.CAMERA_START_TIMEOUT / 1000} seconds`);
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

  // Stats monitoring methods for optimized mode
  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        console.log(`⏭️ Skipping stats update for camera ${cameraId} - system shutting down`);
        return;
      }

      try {
        const statsEndpoint = `/api/video_streaming/video/optimized/stats/${cameraId}`;
        const response = await api.get(statsEndpoint);
        
        const streamStats = response.data.streams?.[0];
        
        if (streamStats && this.detectionService.currentStreams.has(cameraId)) {
          const currentStats = this.detectionService.detectionStats.get(streamKey) || {};
          
          const updatedStats = {
            ...currentStats,
            objectDetected: this.determineDetectionStatus(streamStats),
            detectionCount: streamStats.detections_processed || currentStats.detectionCount,
            nonTargetCount: streamStats.non_target_count || currentStats.nonTargetCount,
            lastDetectionTime: streamStats.last_detection,
            avgProcessingTime: streamStats.avg_detection_time_ms || streamStats.avg_processing_time || 0,
            streamQuality: streamStats.current_quality || currentStats.streamQuality,
            detectionFps: streamStats.detection_interval ? (25 / streamStats.detection_interval) : currentStats.detectionFps,
            queueDepth: streamStats.detection_backlog || 0,
            isStreamActive: streamStats.is_active,
            mode: 'optimized'
          };

          this.detectionService.detectionStats.set(streamKey, updatedStats);
          this.notifyStatsListeners(cameraId, updatedStats);
        }
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error polling optimized detection stats:", error);
        }
      }
    }, 2000);

    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { pollInterval, listeners: [] });
    } else {
      this.detectionService.eventListeners.get(cameraId).pollInterval = pollInterval;
    }
  };

  stopStatsMonitoring = (cameraId) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData?.pollInterval) {
      clearInterval(eventData.pollInterval);
    }
    this.detectionService.eventListeners.delete(cameraId);
  };

  addStatsListener = (cameraId, callback) => {
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { listeners: [] });
    }
    this.detectionService.eventListeners.get(cameraId).listeners.push(callback);
  };

  removeStatsListener = (cameraId, callback) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData) {
      eventData.listeners = eventData.listeners.filter(cb => cb !== callback);
    }
  };

  notifyStatsListeners = (cameraId, stats) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData?.listeners) {
      eventData.listeners.forEach(callback => {
        try {
          callback(stats);
        } catch (error) {
          console.error("Error in optimized stats listener callback:", error);
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
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream?.streamKey) {
      return this.detectionService.detectionStats.get(stream.streamKey) || {
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        mode: 'optimized'
      };
    }
    return null;
  };

  async getAllOptimizedStreamingStats() {
    try {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        return {
          active_streams: 0,
          avg_processing_time_ms: 0,
          total_detections: 0,
          system_load_percent: 0,
          memory_usage_mb: 0,
          mode: 'optimized'
        };
      }

      const statsEndpoint = '/api/video_streaming/video/optimized/stats';
      const response = await api.get(statsEndpoint);
      
      return { ...response.data, mode: 'optimized' };
    } catch (error) { 
      console.error("❌ Error getting optimized streaming stats:", error);
      throw error;
    }
  }

  async getOptimizedPerformanceComparison() {
    try {
      const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
      return response.data;
    } catch (error) {
      console.error("❌ Error getting optimized performance comparison:", error);
      throw error;
    }
  };

  async stopAllOptimizedStreams() {
    try {
      console.log('🛑 Stopping all optimized streams...');
      
      const optimizedStreams = Array.from(this.detectionService.currentStreams.entries())
        .filter(([_, stream]) => stream.type && stream.type.includes('optimized'));
      
      const stopPromises = optimizedStreams.map(([cameraId, _]) => 
        this.stopOptimizedDetectionFeed(cameraId, false)
      );
      
      await Promise.allSettled(stopPromises);
      
      // Stop all optimized streams on API
      try {
        await api.post('/api/video_streaming/video/optimized/streams/stop_all');
      } catch (error) {
        console.warn('⚠️ Error calling optimized stop_all API:', error.message);
      }
      
      console.log("✅ Stopped all optimized streams");
    } catch (error) {
      console.error("❌ Error stopping all optimized streams:", error);
      throw error;
    }
  }

  cleanup = async () => { 
    try {
      console.log('🧹 Starting OptimizedStreamManager cleanup...');
      await this.stopAllOptimizedStreams();
      console.log('✅ OptimizedStreamManager cleanup completed');
    } catch (error) {
      console.error("❌ Error during OptimizedStreamManager cleanup:", error);
    }
  };
}