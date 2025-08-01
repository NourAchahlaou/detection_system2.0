import api from "../../../utils/UseAxios";

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// Performance modes based on system capabilities
const PerformanceModes = {
  BASIC: 'basic',
  STANDARD: 'standard', 
  ENHANCED: 'enhanced',
  HIGH_PERFORMANCE: 'high_performance'
};

// Streaming types
const StreamingTypes = {
  BASIC: 'basic',
  OPTIMIZED: 'optimized'
};

export class StreamManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  // ===================
  // FREEZE/UNFREEZE METHODS
  // ===================

  async freezeStream(cameraId) {
    try {
      console.log(`🧊 Freezing stream for camera ${cameraId}...`);
      
      const response = await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/freeze`);
      
      if (response.data.status === 'frozen') {
        const freezeInfo = {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen'
        };
        
        this.detectionService.frozenStreams.set(cameraId, freezeInfo);
        this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        
        console.log(`✅ Stream frozen for camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to freeze stream');
      }
    } catch (error) {
      console.error(`❌ Error freezing stream for camera ${cameraId}:`, error);
      throw new Error(`Failed to freeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async unfreezeStream(cameraId) {
    try {
      console.log(`🔥 Unfreezing stream for camera ${cameraId}...`);
      
      const response = await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/unfreeze`);
      
      if (response.data.status === 'unfrozen') {
        this.detectionService.frozenStreams.delete(cameraId);
        this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
        
        console.log(`✅ Stream unfrozen for camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to unfreeze stream');
      }
    } catch (error) {
      console.error(`❌ Error unfreezing stream for camera ${cameraId}:`, error);
      throw new Error(`Failed to unfreeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getStreamFreezeStatus(cameraId) {
    try {
      const response = await api.get(`/api/video_streaming/video/basic/stream/${cameraId}/status`);
      
      const isFrozen = response.data.is_frozen || false;
      const streamActive = response.data.stream_active || false;
      
      if (isFrozen && !this.detectionService.frozenStreams.has(cameraId)) {
        // Update local state
        this.detectionService.frozenStreams.set(cameraId, {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen'
        });
      } else if (!isFrozen && this.detectionService.frozenStreams.has(cameraId)) {
        this.detectionService.frozenStreams.delete(cameraId);
      }
      
      return {
        cameraId: parseInt(cameraId),
        isFrozen,
        streamActive,
        ...response.data
      };
    } catch (error) {
      console.error(`❌ Error getting freeze status for camera ${cameraId}:`, error);
      return {
        cameraId: parseInt(cameraId),
        isFrozen: false,
        streamActive: false,
        error: error.message
      };
    }
  }

  // ===================
  // DETECTION METHODS
  // ===================

  async performOnDemandDetection(cameraId, targetLabel, options = {}) {
    try {
      const { quality = 85, autoUnfreeze = false, unfreezeDelay = 2.0 } = options;
      
      console.log(`🎯 Performing on-demand detection for camera ${cameraId}, target: '${targetLabel}'`);
      
      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      // Choose detection endpoint based on auto-unfreeze option
      const endpoint = autoUnfreeze 
        ? `/api/detection/basic/detect/${cameraId}/with-unfreeze`
        : `/api/detection/basic/detect/${cameraId}`;

      const requestBody = {
        target_label: targetLabel,
        quality: quality
      };

      if (autoUnfreeze) {
        requestBody.unfreeze_delay = unfreezeDelay;
      }

      const response = await api.post(endpoint, requestBody);

      if (response.data.success) {
        const detectionData = response.data.data;
        
        // Update freeze status
        if (detectionData.stream_frozen && !autoUnfreeze) {
          this.detectionService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            detectionPerformed: true
          });
          this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        } else if (autoUnfreeze) {
          this.detectionService.frozenStreams.delete(cameraId);
          this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
        }

        // Update detection stats
        const streamKey = `basic_${cameraId}_${targetLabel}`;
        const updatedStats = {
          objectDetected: detectionData.detected_target,
          detectionCount: detectionData.detected_target ? 1 : 0,
          lastDetectionTime: detectionData.detected_target ? Date.now() : null,
          avgProcessingTime: detectionData.processing_time_ms,
          confidence: detectionData.confidence,
          mode: 'basic_on_demand',
          streamFrozen: detectionData.stream_frozen,
          autoUnfrozen: autoUnfreeze
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ On-demand detection completed for camera ${cameraId}${autoUnfreeze ? ' (auto-unfrozen)' : ' (stream frozen)'}`);
        
        return {
          success: true,
          detected: detectionData.detected_target,
          confidence: detectionData.confidence,
          processingTime: detectionData.processing_time_ms,
          frameWithOverlay: detectionData.frame_with_overlay,
          streamFrozen: detectionData.stream_frozen,
          autoUnfrozen: autoUnfreeze,
          timestamp: detectionData.timestamp
        };
      } else {
        throw new Error('Detection request failed');
      }

    } catch (error) {
      console.error("❌ Error performing on-demand detection:", error);
      throw new Error(`Failed to perform detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  async performBatchDetection(detections = []) {
    try {
      console.log(`🎯 Performing batch detection on ${detections.length} streams`);

      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform batch detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      if (detections.length === 0) {
        throw new Error('No detections provided');
      }

      if (detections.length > 3) {
        throw new Error('Maximum 3 detections per batch in basic mode');
      }

      const response = await api.post('/api/detection/basic/detect/batch', {
        detections: detections
      });

      if (response.data.success) {
        const results = response.data.results;
        
        // Update local state for each detection
        results.forEach(result => {
          if (result.success && result.data) {
            const cameraId = result.camera_id;
            const detectionData = result.data;
            
            // Update freeze status
            if (detectionData.stream_frozen && !result.auto_unfrozen) {
              this.detectionService.frozenStreams.set(cameraId, {
                cameraId: parseInt(cameraId),
                frozenAt: Date.now(),
                status: 'frozen',
                detectionPerformed: true
              });
              this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
            } else if (result.auto_unfrozen) {
              this.detectionService.frozenStreams.delete(cameraId);
              this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
            }

            // Update detection stats
            const streamKey = `basic_${cameraId}_batch`;
            const updatedStats = {
              objectDetected: detectionData.detected_target,
              detectionCount: detectionData.detected_target ? 1 : 0,
              lastDetectionTime: detectionData.detected_target ? Date.now() : null,
              avgProcessingTime: detectionData.processing_time_ms,
              confidence: detectionData.confidence,
              mode: 'basic_batch',
              streamFrozen: detectionData.stream_frozen,
              autoUnfrozen: result.auto_unfrozen
            };

            this.detectionService.detectionStats.set(streamKey, updatedStats);
            this.notifyStatsListeners(cameraId, updatedStats);
          }
        });

        console.log(`✅ Batch detection completed: ${response.data.successful}/${response.data.total_processed} successful`);
        return response.data;
      } else {
        throw new Error('Batch detection request failed');
      }

    } catch (error) {
      console.error("❌ Error performing batch detection:", error);
      throw new Error(`Failed to perform batch detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // ENHANCED STREAMING METHODS WITH AUTO MODE
  // ===================

  async startDetectionFeedWithAutoMode(cameraId, targetLabel, options = {}) {
    try {
      console.log(`🎯 Starting detection feed with auto mode for camera ${cameraId}`);
      
      // Update system profile if needed
      if (!this.detectionService.systemProfile) {
        await this.detectionService.updateSystemProfile();
      }

      // Check if we should use optimized mode
      if (this.detectionService.shouldUseOptimizedMode()) {
        console.log(`🎯 Attempting optimized detection for camera ${cameraId}`);
        
        // Ensure detection service is ready before starting optimized mode
        const readinessCheck = await this.detectionService.ensureDetectionServiceReady();
        
        if (readinessCheck.success) {
          try {
            return await this.startOptimizedDetectionFeed(cameraId, targetLabel, options);
          } catch (optimizedError) {
            console.log(`⚠️ Optimized mode failed: ${optimizedError.message}, falling back to basic`);
            // Force switch to basic mode
            await this.detectionService.switchToBasicMode();
            return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
          }
        } else {
          console.log('🎯 Detection service not ready, using basic detection');
          // Ensure we're in basic mode
          if (!this.detectionService.shouldUseBasicMode()) {
            await this.detectionService.switchToBasicMode();
          }
          return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
        }
      } else {
        console.log(`🎯 Using basic detection for camera ${cameraId} (Performance mode: ${this.detectionService.currentPerformanceMode})`);
        return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
      }
      
    } catch (error) {
      console.error('❌ Error starting detection feed with auto mode:', error);
      
      // Final fallback to basic mode
      console.log('⚠️ Final fallback to basic mode due to error');
      try {
        await this.detectionService.switchToBasicMode();
        return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
      } catch (basicError) {
        throw new Error(`All detection modes failed: ${basicError.message}`);
      }
    }
  }


  async startStreamWithAutoMode(cameraId, options = {}) {
    try {
      // Update system profile if needed
      if (!this.detectionService.systemProfile) {
        await this.detectionService.updateSystemProfile();
      }

      if (this.detectionService.shouldUseBasicMode()) {
        console.log(`📺 Using basic streaming for camera ${cameraId} (Performance mode: ${this.detectionService.currentPerformanceMode})`);
        return await this.startBasicStream(cameraId, options);
      } else {
        console.log(`📺 Using optimized streaming for camera ${cameraId} (Performance mode: ${this.detectionService.currentPerformanceMode})`);
        return await this.startOptimizedStream(cameraId, options);
      }
    } catch (error) {
      console.error('❌ Error starting stream with auto mode:', error);
      
      // Fallback to basic mode on error
      if (!this.detectionService.shouldUseBasicMode()) {
        console.log('⚠️ Falling back to basic mode due to error');
        return await this.startBasicStream(cameraId, options);
      }
      
      throw error;
    }
  }

  // ===================
  // BASIC STREAMING METHODS
  // ===================

  async startBasicDetectionFeed(cameraId, targetLabel, options = {}) {
    try {
      console.log(`🎯 Starting basic detection feed for camera ${cameraId} with target: ${targetLabel}`);

      if (!this.detectionService.canStart()) {
        throw new Error(`Cannot start detection in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      // Start camera
      await this.ensureCameraStarted(cameraId);

      // Store stream info
      const streamKey = `basic_${cameraId}_${targetLabel}`;
      this.detectionService.currentStreams.set(cameraId, {
        url: `/api/video_streaming/video/basic/stream/${cameraId}`,
        targetLabel,
        streamKey,
        startTime: Date.now(),
        isActive: true,
        type: 'basic_detection'
      });

      // Update state to running
      this.detectionService.setState(DetectionStates.RUNNING, `Started basic detection for camera ${cameraId}`);

      // Start detection polling for basic mode (but less frequently since it's on-demand)
      this.startBasicDetectionMonitoring(cameraId, targetLabel);

      console.log(`✅ Successfully started basic detection feed for camera ${cameraId}`);
      return `/api/video_streaming/video/basic/stream/${cameraId}`;

    } catch (error) {
      console.error("❌ Error starting basic detection feed:", error);
      throw new Error(`Failed to start basic detection feed: ${error.message}`);
    }
  }

  async startBasicStream(cameraId, options = {}) {
    try {
      const { streamQuality = 85 } = options;
      console.log(`📺 Starting basic stream for camera ${cameraId}`);

      // Fix: Use explicit state check instead of canStart()
      if (this.detectionService.state !== DetectionStates.READY) {
        throw new Error(`Cannot start stream in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      await this.ensureCameraStarted(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/basic/stream/${cameraId}?${params}`;
      
      this.detectionService.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `basic_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'basic_stream'
      });

      if (this.detectionService.state === DetectionStates.READY) {
        this.detectionService.setState(DetectionStates.RUNNING, `Started basic stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started basic stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting basic stream:", error);
      throw new Error(`Failed to start basic stream: ${error.message}`);
    }
  }

  async stopBasicStream(cameraId) {
    try {
      const stream = this.detectionService.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.startsWith('basic')) {
        console.log(`⏹️ Stopping basic stream for camera ${cameraId}`);
        
        // Unfreeze stream if it's frozen
        if (this.detectionService.isStreamFrozen(cameraId)) {
          try {
            await this.unfreezeStream(cameraId);
          } catch (error) {
            console.warn(`⚠️ Error unfreezing stream during stop for camera ${cameraId}:`, error.message);
          }
        }
        
        this.stopStatsMonitoring(cameraId);
        
        try {
          await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/stop`);
          await api.post("/api/artifact_keeper/camera/stop");
        } catch (error) {
          console.warn(`⚠️ Error stopping basic stream API for camera ${cameraId}:`, error.message);
        }
        
        this.detectionService.currentStreams.delete(cameraId);
        
        // Update state based on remaining streams
        if (this.detectionService.currentStreams.size === 0 && this.detectionService.state === DetectionStates.RUNNING) {
          this.detectionService.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        console.log(`✅ Successfully stopped basic stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping basic stream:", error);
      throw error;
    }
  }

  async startBasicDetectionMonitoring(cameraId, targetLabel) {
    // For basic mode, we don't need continuous polling since detection is on-demand
    // Just set up monitoring for stream status and freeze state
    const monitorInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        clearInterval(monitorInterval);
        return;
      }

      try {
        // Check stream status and freeze state
        const freezeStatus = await this.getStreamFreezeStatus(cameraId);
        
        // Update stats based on stream status
        const streamKey = `basic_${cameraId}_${targetLabel}`;
        const currentStats = this.detectionService.detectionStats.get(streamKey) || {};
        
        const updatedStats = {
          ...currentStats,
          streamActive: freezeStatus.streamActive,
          isFrozen: freezeStatus.isFrozen,
          lastStatusCheck: Date.now(),
          mode: 'basic_monitoring'
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);
        
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error in basic detection monitoring:", error);
        }
      }
    }, 5000); // Check every 5 seconds (less frequent than continuous detection)

    // Store interval for cleanup
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { monitorInterval, listeners: [] });
    } else {
      this.detectionService.eventListeners.get(cameraId).monitorInterval = monitorInterval;
    }
  }

  async startBasicDetectionPolling(cameraId, targetLabel) {
    const pollInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        clearInterval(pollInterval);
        return;
      }

      try {
        // Perform detection on single frame
        const response = await api.post(`/api/detection/basic/detect/${cameraId}`, {
          target_label: targetLabel,
          quality: 85
        });

        if (response.data.success) {
          const detectionData = response.data.data;
          const streamKey = `basic_${cameraId}_${targetLabel}`;
          
          // Update detection stats
          const currentStats = this.detectionService.detectionStats.get(streamKey) || {};
          const updatedStats = {
            ...currentStats,
            objectDetected: detectionData.target_detected,
            detectionCount: detectionData.target_detected ? (currentStats.detectionCount || 0) + 1 : (currentStats.detectionCount || 0),
            lastDetectionTime: detectionData.target_detected ? Date.now() : currentStats.lastDetectionTime,
            avgProcessingTime: detectionData.processing_time_ms,
            confidence: detectionData.confidence
          };

          this.detectionService.detectionStats.set(streamKey, updatedStats);
          this.notifyStatsListeners(cameraId, updatedStats);
        }
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error in basic detection polling:", error);
        }
      }
    }, 2000); // Poll every 2 seconds for basic mode

    // Store interval for cleanup
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { pollInterval, listeners: [] });
    } else {
      this.detectionService.eventListeners.get(cameraId).pollInterval = pollInterval;
    }
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
        console.log('⚠️ System in basic mode, falling back to basic stream');
        return await this.startBasicStream(cameraId, options);
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

async stopAllStreams(performShutdown = true) {
    try {
      console.log('🛑 Stopping all streams...');
      
      const stopPromises = Array.from(this.detectionService.currentStreams.keys()).map(cameraId => {
        const stream = this.detectionService.currentStreams.get(cameraId);
        if (stream.type && stream.type.startsWith('basic')) {
          return this.stopBasicStream(cameraId);
        } else {
          return this.stopOptimizedDetectionFeed(cameraId, false);
        }
      });
      
      await Promise.allSettled(stopPromises);
      
      // Stop all streams on both endpoints
      try {
        if (this.detectionService.shouldUseOptimizedMode()) {
          await api.post('/api/video_streaming/video/optimized/streams/stop_all');
        }
       
      } catch (error) {
        console.warn('⚠️ Error calling stop_all APIs:', error.message);
      }
      
      // Unfreeze all frozen streams in basic mode
      if (this.detectionService.shouldUseBasicMode()) {
        const frozenStreams = this.detectionService.getFrozenStreams();
        for (const frozenStream of frozenStreams) {
          try {
            await this.unfreezeStream(frozenStream.cameraId);
          } catch (error) {
            console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId}:`, error.message);
          }
        }
      }
      
      this.detectionService.currentStreams.clear();
      this.detectionService.detectionStats.clear();
      for (const cameraId of this.detectionService.eventListeners.keys()) {
        this.stopStatsMonitoring(cameraId);
      }
      
      if (this.detectionService.state === DetectionStates.RUNNING) {
        this.detectionService.setState(DetectionStates.READY, 'All streams stopped');
      }
      
      if (performShutdown) {
        try {
          console.log('🔄 Performing graceful detection shutdown after stopping all streams...');
          await this.detectionService.gracefulShutdown();
          console.log('✅ All streams stopped and detection service gracefully shut down');
        } catch (shutdownError) {
          console.error('⚠️ Graceful shutdown failed, but all streams stopped:', shutdownError.message);
        }
      }
      
      console.log("✅ Stopped all streams");
    } catch (error) {
      console.error("❌ Error stopping all streams:", error);
      throw error;
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
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.CAMERA_START_TIMEOUT);

      try {
        const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
          camera_id: numericCameraId
        }, {
          signal: controller.signal,
          timeout: this.detectionService.CAMERA_START_TIMEOUT
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

  // Stats monitoring methods (enhanced for mode awareness)
  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        console.log(`⏭️ Skipping stats update for camera ${cameraId} - system shutting down`);
        return;
      }

      try {
        let statsEndpoint = `/api/video_streaming/video/optimized/stats/${cameraId}`;
        if (this.detectionService.shouldUseBasicMode()) {
          statsEndpoint = `/api/video_streaming/video/basic/stats`;
        }

        const response = await api.get(statsEndpoint);
        
        let streamStats;
        if (this.detectionService.shouldUseBasicMode()) {
          // Basic mode returns different stats format
          streamStats = response.data.stream_stats?.find(s => s.camera_id === cameraId);
          
          // Also get freeze status for basic mode
          const freezeStatus = await this.getStreamFreezeStatus(cameraId);
          if (streamStats) {
            streamStats.is_frozen = freezeStatus.isFrozen;
            streamStats.stream_active = freezeStatus.streamActive;
          }
        } else {
          streamStats = response.data.streams?.[0];
        }
        
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
            isFrozen: streamStats.is_frozen || false,
            mode: this.detectionService.currentStreamingType
          };

          this.detectionService.detectionStats.set(streamKey, updatedStats);
          this.notifyStatsListeners(cameraId, updatedStats);
        }
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error polling detection stats:", error);
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
    if (eventData?.monitorInterval) {
      clearInterval(eventData.monitorInterval);
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
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream?.streamKey) {
      return this.detectionService.detectionStats.get(stream.streamKey) || {
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0,
        lastDetectionTime: null,
        avgProcessingTime: 0,
        isFrozen: this.detectionService.isStreamFrozen(cameraId),
        mode: this.detectionService.currentStreamingType
      };
    }
    return null;
  };

  async getAllStreamingStats() {
    try {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        return {
          active_streams: 0,
          avg_processing_time_ms: 0,
          total_detections: 0,
          system_load_percent: 0,
          memory_usage_mb: 0,
          frozen_streams: this.detectionService.getFrozenStreams().length,
          mode: this.detectionService.currentStreamingType
        };
      }

      let statsEndpoint = '/api/video_streaming/video/optimized/stats';
      if (this.detectionService.shouldUseBasicMode()) {
        statsEndpoint = '/api/video_streaming/video/basic/stats';
      }

      const response = await api.get(statsEndpoint);
      
      // Add freeze information for basic mode
      if (this.detectionService.shouldUseBasicMode()) {
        response.data.frozen_streams = this.detectionService.getFrozenStreams().length;
        response.data.live_streams = (response.data.total_active_streams || 0) - response.data.frozen_streams;
      }
      
      return { ...response.data, mode: this.detectionService.currentStreamingType };
    } catch (error) { 
      console.error("❌ Error getting streaming stats:", error);
      throw error;
    }
  }

  getPerformanceComparison = async () => {
    try {
      // Get comparison from optimized endpoint if available
      if (this.detectionService.shouldUseOptimizedMode()) {
        const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
        return response.data;
      } else {
        // Return basic performance metrics for basic mode
        const basicStats = await this.getAllStreamingStats();
        return {
          current_mode: 'basic',
          basic_performance: basicStats,
          optimized_performance: null,
          recommendation: 'Basic mode - suitable for current system specifications',
          freeze_capability: true,
          on_demand_detection: true
        };
      }
    } catch (error) {
      console.error("❌ Error getting performance comparison:", error);
      throw error;
    }
  };

  // Legacy method compatibility
  stopDetectionFeed = async (cameraId, performShutdown = true) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream) {
      if (stream.type && stream.type.startsWith('basic')) {
        return this.stopBasicStream(cameraId);
      } else {
        return this.stopOptimizedDetectionFeed(cameraId, performShutdown);
      }
    }
  };

  cleanup = async () => { 
    try {
      console.log('🧹 Starting StreamManager cleanup...');
      await this.stopAllStreams(true);
      console.log('✅ StreamManager cleanup completed');
    } catch (error) {
      console.error("❌ Error during StreamManager cleanup:", error);
    }
  };
}