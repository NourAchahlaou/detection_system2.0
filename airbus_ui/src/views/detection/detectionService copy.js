import api from "../../utils/UseAxios";

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
    
    // System profiling and performance mode
    this.systemProfile = null;
    this.currentPerformanceMode = PerformanceModes.BASIC; // Default to basic
    this.currentStreamingType = StreamingTypes.BASIC; // Default to basic
    this.systemCapabilities = null;
    this.profileUpdateListeners = new Set();
    this.autoModeEnabled = true; // Automatically switch modes based on specs
    
    // Basic mode detection state tracking
    this.basicModeDetectionState = new Map(); // Track detection state per camera
    this.onDemandDetectionCallbacks = new Map(); // Store callbacks for detection results
    
    // Initialization tracking
    this.initializationPromise = null;
    this.hasPerformedInitialHealthCheck = false;
    this.hasPerformedPostShutdownCheck = false;
    
    // Timeouts
    this.INITIALIZATION_TIMEOUT = 30000;
    this.HEALTH_CHECK_TIMEOUT = 10000;
    this.CAMERA_START_TIMEOUT = 15000;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.HEALTH_CHECK_COOLDOWN = 5000;
    this.SYSTEM_PROFILE_CACHE_DURATION = 30000; // 30 seconds cache
    this.DETECTION_TIMEOUT = 15000; // Timeout for on-demand detection
    
    console.log('🔧 DetectionService initialized with state:', this.state);
    
    // Initialize system profiling
    this.initializeSystemProfiling();
  }

  // ===================
  // SYSTEM PROFILING METHODS (keeping existing)
  // ===================

  async initializeSystemProfiling() {
    try {
      console.log('🖥️ Initializing system profiling...');
      await this.updateSystemProfile();
      console.log(`✅ System profiling initialized - Mode: ${this.currentPerformanceMode}, Streaming: ${this.currentStreamingType}`);
    } catch (error) {
      console.error('⚠️ Failed to initialize system profiling:', error);
      // Continue with basic mode as fallback
      this.currentPerformanceMode = PerformanceModes.BASIC;
      this.currentStreamingType = StreamingTypes.BASIC;
    }
  }

  async updateSystemProfile(forceRefresh = false) {
    try {
      // Check cache first
      if (!forceRefresh && this.systemProfile && 
          this.systemProfile.timestamp && 
          (Date.now() - new Date(this.systemProfile.timestamp).getTime()) < this.SYSTEM_PROFILE_CACHE_DURATION) {
        console.log('📊 Using cached system profile');
        return this.systemProfile;
      }

      console.log('🔍 Fetching system profile from artifact keeper...');
      
      // Get system profile from artifact keeper
      const profileResponse = await api.get('/api/artifact_keeper/system/profile', {
        params: { force_refresh: forceRefresh }
      });
      
      this.systemProfile = profileResponse.data;
      
      // Get performance recommendation
      const recommendationResponse = await api.get('/api/artifact_keeper/system/performance/recommendation');
      const recommendation = recommendationResponse.data;
      
      // Get system capabilities
      const capabilitiesResponse = await api.get('/api/artifact_keeper/system/capabilities');
      this.systemCapabilities = capabilitiesResponse.data;
      
      // Determine optimal performance mode and streaming type
      this.determineOptimalSettings(this.systemProfile, recommendation, this.systemCapabilities);
      
      // Notify listeners
      this.notifyProfileUpdateListeners();
      
      console.log(`📊 System profile updated - Performance: ${this.currentPerformanceMode}, Streaming: ${this.currentStreamingType}`);
      console.log(`   CPU: ${this.systemProfile.cpu_cores} cores @ ${this.systemProfile.cpu_frequency_mhz}MHz`);
      console.log(`   Memory: ${this.systemProfile.available_memory_gb}GB available`);
      console.log(`   GPU: ${this.systemProfile.gpu_available ? this.systemProfile.gpu_name : 'None'}`);
      console.log(`   Score: ${this.systemProfile.performance_score}/100`);
      
      return this.systemProfile;
      
    } catch (error) {
      console.error('❌ Error updating system profile:', error);
      
      // Fallback to basic mode if profiling fails
      if (!this.systemProfile) {
        this.currentPerformanceMode = PerformanceModes.BASIC;
        this.currentStreamingType = StreamingTypes.BASIC;
        console.log('⚠️ Using fallback basic mode due to profiling error');
      }
      
      throw error;
    }
  }

  determineOptimalSettings(profile, recommendation, capabilities) {
    try {
      const score = profile.performance_score || 0;
      const recommendedMode = recommendation.final_recommendation || 'basic';
      const meetsMinimum = profile.meets_minimum_requirements || false;
      const gpuAvailable = profile.gpu_available || false;
      const cudaAvailable = profile.cuda_available || false;
      
      // Determine performance mode based on system specs
      if (score >= 80 && gpuAvailable && cudaAvailable && meetsMinimum) {
        this.currentPerformanceMode = PerformanceModes.HIGH_PERFORMANCE;
        this.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else if (score >= 60 && gpuAvailable && meetsMinimum) {
        this.currentPerformanceMode = PerformanceModes.ENHANCED;
        this.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else if (score >= 40 && meetsMinimum) {
        this.currentPerformanceMode = PerformanceModes.STANDARD;
        this.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else {
        this.currentPerformanceMode = PerformanceModes.BASIC;
        this.currentStreamingType = StreamingTypes.BASIC;
      }
      
      // Override with explicit recommendation if available
      if (recommendedMode === 'basic') {
        this.currentPerformanceMode = PerformanceModes.BASIC;
        this.currentStreamingType = StreamingTypes.BASIC;
      } else if (recommendedMode === 'standard' && this.currentPerformanceMode === PerformanceModes.BASIC) {
        this.currentPerformanceMode = PerformanceModes.STANDARD;
        this.currentStreamingType = StreamingTypes.OPTIMIZED;
      }
      
      console.log(`🎯 Determined optimal settings based on score ${score}:`);
      console.log(`   Performance Mode: ${this.currentPerformanceMode}`);
      console.log(`   Streaming Type: ${this.currentStreamingType}`);
      console.log(`   Recommended Mode: ${recommendedMode}`);
      
    } catch (error) {
      console.error('❌ Error determining optimal settings:', error);
      // Fallback to basic
      this.currentPerformanceMode = PerformanceModes.BASIC;
      this.currentStreamingType = StreamingTypes.BASIC;
    }
  }

  // ===================
  // ENHANCED BASIC MODE METHODS FOR REDIS COORDINATION
  // ===================

  async startBasicStreamOnly(cameraId, options = {}) {
    try {
      const { streamQuality = 85 } = options;
      console.log(`📺 Starting Redis basic stream (no detection) for camera ${cameraId}`);

      if (!this.canStart()) {
        throw new Error(`Cannot start stream in state: ${this.state}. Current state must be READY.`);
      }

      await this.ensureCameraStarted(cameraId);

      // Use the Redis basic streaming endpoint
      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/redis-basic/stream/${cameraId}?${params}`;
      
      this.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `redis_basic_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'redis_basic_stream',
        streamQuality: streamQuality
      });

      // Initialize basic mode detection state
      this.basicModeDetectionState.set(cameraId, {
        isFrozen: false,
        lastDetectionTime: null,
        detectionInProgress: false,
        frozenFrame: null
      });

      if (this.state === DetectionStates.READY) {
        this.setState(DetectionStates.RUNNING, `Started Redis basic stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started Redis basic stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting Redis basic stream:", error);
      throw new Error(`Failed to start Redis basic stream: ${error.message}`);
    }
  }

  async performOnDemandDetection(cameraId, targetLabel, options = {}) {
    try {
      const { quality = 85, autoUnfreeze = false, unfreezeDelay = 2.0 } = options;
      
      console.log(`🎯 Performing on-demand detection for camera ${cameraId}, target: ${targetLabel}`);

      // Check if camera stream exists
      const stream = this.currentStreams.get(cameraId);
      if (!stream || !stream.type.includes('redis_basic')) {
        throw new Error(`No active Redis basic stream found for camera ${cameraId}. Start stream first.`);
      }

      // Check if detection is already in progress
      const detectionState = this.basicModeDetectionState.get(cameraId);
      if (detectionState?.detectionInProgress) {
        throw new Error(`Detection already in progress for camera ${cameraId}`);
      }

      // Update detection state
      this.basicModeDetectionState.set(cameraId, {
        ...detectionState,
        detectionInProgress: true,
        lastDetectionTime: Date.now()
      });

      // Create timeout for detection
      const detectionPromise = new Promise(async (resolve, reject) => {
        const timeoutId = setTimeout(() => {
          reject(new Error('Detection request timed out'));
        }, this.DETECTION_TIMEOUT);

        try {
          let detectionResponse;
          
          if (autoUnfreeze) {
            // Use the auto-unfreeze endpoint
            detectionResponse = await api.post(`/api/detection/redis-basic/detect/${cameraId}/with-unfreeze`, {
              target_label: targetLabel,
              quality: quality,
              unfreeze_delay: unfreezeDelay
            });
          } else {
            // Use the standard detection endpoint (keeps stream frozen)
            detectionResponse = await api.post(`/api/detection/redis-basic/detect/${cameraId}`, {
              target_label: targetLabel,
              quality: quality
            });
          }

          clearTimeout(timeoutId);

          if (detectionResponse.data.success) {
            const detectionData = detectionResponse.data.data;
            
            // Update detection state
            this.basicModeDetectionState.set(cameraId, {
              ...this.basicModeDetectionState.get(cameraId),
              detectionInProgress: false,
              isFrozen: !autoUnfreeze, // Stream is frozen unless auto-unfrozen
              lastDetectionResult: detectionData,
              frozenFrame: detectionData.annotated_image_base64
            });

            // Update detection stats
            this.updateBasicDetectionStats(cameraId, detectionData, targetLabel);

            console.log(`✅ On-demand detection completed for camera ${cameraId}`);
            resolve({
              success: true,
              data: detectionData,
              isFrozen: !autoUnfreeze,
              autoUnfreeze: autoUnfreeze
            });
          } else {
            throw new Error('Detection request failed');
          }
        } catch (error) {
          clearTimeout(timeoutId);
          reject(error);
        }
      });

      return await detectionPromise;

    } catch (error) {
      // Reset detection state on error
      const currentState = this.basicModeDetectionState.get(cameraId);
      if (currentState) {
        this.basicModeDetectionState.set(cameraId, {
          ...currentState,
          detectionInProgress: false
        });
      }
      
      console.error(`❌ Error performing on-demand detection for camera ${cameraId}:`, error);
      throw new Error(`On-demand detection failed: ${error.message}`);
    }
  }

  async unfreezeBasicStream(cameraId) {
    try {
      console.log(`🔥 Unfreezing Redis basic stream for camera ${cameraId}`);

      // Check if stream exists and is frozen
      const stream = this.currentStreams.get(cameraId);
      if (!stream || !stream.type.includes('redis_basic')) {
        throw new Error(`No active Redis basic stream found for camera ${cameraId}`);
      }

      const detectionState = this.basicModeDetectionState.get(cameraId);
      if (!detectionState?.isFrozen) {
        console.log(`ℹ️ Stream for camera ${cameraId} is not frozen`);
        return { success: true, message: 'Stream is not frozen' };
      }

      // Unfreeze via Redis detection service
      const response = await api.post(`/api/detection/redis-basic/stream/${cameraId}/unfreeze`);

      if (response.data.success) {
        // Update detection state
        this.basicModeDetectionState.set(cameraId, {
          ...detectionState,
          isFrozen: false,
          frozenFrame: null
        });

        console.log(`✅ Successfully unfroze Redis basic stream for camera ${cameraId}`);
        return { success: true, message: 'Stream resumed' };
      } else {
        throw new Error('Failed to unfreeze stream via Redis');
      }

    } catch (error) {
      console.error(`❌ Error unfreezing Redis basic stream for camera ${cameraId}:`, error);
      throw new Error(`Failed to unfreeze stream: ${error.message}`);
    }
  }

  async getBasicStreamStatus(cameraId) {
    try {
      const stream = this.currentStreams.get(cameraId);
      const detectionState = this.basicModeDetectionState.get(cameraId);
      
      if (!stream || !stream.type.includes('redis_basic')) {
        return {
          exists: false,
          message: 'No Redis basic stream found'
        };
      }

      // Get status from Redis streaming service
      const streamResponse = await api.get(`/api/video_streaming/video/redis-basic/stream/${cameraId}/status`);
      
      // Get status from Redis detection service
      const detectionResponse = await api.get(`/api/detection/redis-basic/stream/${cameraId}/status`);

      return {
        exists: true,
        stream: {
          url: stream.url,
          startTime: stream.startTime,
          streamQuality: stream.streamQuality,
          type: stream.type,
          isActive: streamResponse.data.stream_active
        },
        detection: {
          isFrozen: streamResponse.data.is_frozen || detectionState?.isFrozen || false,
          detectionInProgress: detectionState?.detectionInProgress || false,
          lastDetectionTime: detectionState?.lastDetectionTime,
          lastDetectionResult: detectionState?.lastDetectionResult
        },
        redis: {
          streamConnected: streamResponse.data.redis_connected,
          detectionConnected: detectionResponse.data.status.redis_connected
        }
      };

    } catch (error) {
      console.error(`❌ Error getting Redis basic stream status for camera ${cameraId}:`, error);
      throw error;
    }
  }

  updateBasicDetectionStats(cameraId, detectionData, targetLabel) {
    const streamKey = `redis_basic_${cameraId}_${targetLabel}`;
    const currentStats = this.detectionStats.get(streamKey) || {
      objectDetected: false,
      detectionCount: 0,
      lastDetectionTime: null,
      avgProcessingTime: 0,
      confidence: 0,
      mode: 'redis_basic'
    };

    const updatedStats = {
      ...currentStats,
      objectDetected: detectionData.target_detected,
      detectionCount: detectionData.target_detected ? currentStats.detectionCount + 1 : currentStats.detectionCount,
      lastDetectionTime: Date.now(),
      avgProcessingTime: detectionData.processing_time_ms,
      confidence: detectionData.confidence,
      boundingBoxes: detectionData.detections,
      frozenFrame: detectionData.annotated_image_base64
    };

    this.detectionStats.set(streamKey, updatedStats);
    this.notifyStatsListeners(cameraId, updatedStats);
  }

  async stopRedisBasicStream(cameraId) {
    try {
      const stream = this.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.includes('redis_basic')) {
        console.log(`⏹️ Stopping Redis basic stream for camera ${cameraId}`);
        
        // Stop stats monitoring
        this.stopStatsMonitoring(cameraId);
        
        try {
          // Stop the Redis basic stream
          await api.post(`/api/video_streaming/video/redis-basic/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping Redis basic stream API for camera ${cameraId}:`, error.message);
        }
        
        // Clean up local state
        this.currentStreams.delete(cameraId);
        this.basicModeDetectionState.delete(cameraId);
        
        // Clean up detection stats
        const statsKeys = Array.from(this.detectionStats.keys()).filter(key => 
          key.startsWith(`redis_basic_${cameraId}_`)
        );
        statsKeys.forEach(key => this.detectionStats.delete(key));
        
        // Update service state
        if (this.currentStreams.size === 0 && this.state === DetectionStates.RUNNING) {
          this.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        console.log(`✅ Successfully stopped Redis basic stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping Redis basic stream:", error);
      throw error;
    }
  }

  // ===================
  // UPDATED BASIC MODE METHODS
  // ===================

  async startBasicDetectionFeed(cameraId, targetLabel, options = {}) {
    try {
      console.log(`🎯 Starting Redis basic detection feed for camera ${cameraId} with target: ${targetLabel}`);

      if (!this.canStart()) {
        throw new Error(`Cannot start detection in state: ${this.state}. Current state must be READY.`);
      }

      // Start the basic stream first (no detection, just streaming)
      const streamUrl = await this.startBasicStreamOnly(cameraId, options);

      // Store additional detection info
      const stream = this.currentStreams.get(cameraId);
      this.currentStreams.set(cameraId, {
        ...stream,
        targetLabel: targetLabel,
        type: 'redis_basic_detection'
      });

      console.log(`✅ Successfully started Redis basic detection feed for camera ${cameraId} (on-demand mode)`);
      console.log(`💡 Stream is live - use performOnDemandDetection() to detect objects`);
      
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting Redis basic detection feed:", error);
      throw new Error(`Failed to start Redis basic detection feed: ${error.message}`);
    }
  }

  async startBasicStream(cameraId, options = {}) {
    // Use the new Redis basic stream method
    return await this.startBasicStreamOnly(cameraId, options);
  }

  async stopBasicStream(cameraId) {
    // Use the new Redis basic stream stop method
    return await this.stopRedisBasicStream(cameraId);
  }

  // Remove the old basic detection polling since we're now using on-demand
  async startBasicDetectionPolling(cameraId, targetLabel) {
    // This method is no longer needed for Redis basic mode
    // Detection is now on-demand only
    console.log(`ℹ️ Redis basic mode uses on-demand detection - polling not needed for camera ${cameraId}`);
  }

  // ===================
  // ENHANCED STREAMING METHODS WITH REDIS BASIC MODE
  // ===================

  async startDetectionFeedWithAutoMode(cameraId, targetLabel, options = {}) {
    try {
      // Update system profile if needed
      if (!this.systemProfile) {
        await this.updateSystemProfile();
      }

      if (this.shouldUseBasicMode()) {
        console.log(`🎯 Using Redis basic detection for camera ${cameraId} (Performance mode: ${this.currentPerformanceMode})`);
        return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
      } else {
        console.log(`🎯 Using optimized detection for camera ${cameraId} (Performance mode: ${this.currentPerformanceMode})`);
        return await this.startOptimizedDetectionFeed(cameraId, targetLabel, options);
      }
    } catch (error) {
      console.error('❌ Error starting detection feed with auto mode:', error);
      
      // Fallback to basic mode on error
      if (!this.shouldUseBasicMode()) {
        console.log('⚠️ Falling back to Redis basic mode due to error');
        return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
      }
      
      throw error;
    }
  }

  async startStreamWithAutoMode(cameraId, options = {}) {
    try {
      // Update system profile if needed
      if (!this.systemProfile) {
        await this.updateSystemProfile();
      }

      if (this.shouldUseBasicMode()) {
        console.log(`📺 Using Redis basic streaming for camera ${cameraId} (Performance mode: ${this.currentPerformanceMode})`);
        return await this.startBasicStream(cameraId, options);
      } else {
        console.log(`📺 Using optimized streaming for camera ${cameraId} (Performance mode: ${this.currentPerformanceMode})`);
        return await this.startOptimizedStream(cameraId, options);
      }
    } catch (error) {
      console.error('❌ Error starting stream with auto mode:', error);
      
      // Fallback to basic mode on error
      if (!this.shouldUseBasicMode()) {
        console.log('⚠️ Falling back to Redis basic mode due to error');
        return await this.startBasicStream(cameraId, options);
      }
      
      throw error;
    }
  }

  // ===================
  // DETECTION CONVENIENCE METHODS
  // ===================

  // Convenience method for UI components
  async detectObject(cameraId, targetLabel, options = {}) {
    return await this.performOnDemandDetection(cameraId, targetLabel, options);
  }

  // Convenience method for detect and unfreeze
  async detectAndResume(cameraId, targetLabel, options = {}) {
    const { unfreezeDelay = 3.0, ...detectOptions } = options;
    return await this.performOnDemandDetection(cameraId, targetLabel, {
      ...detectOptions,
      autoUnfreeze: true,
      unfreezeDelay: unfreezeDelay
    });
  }

  // Check if camera stream is frozen
  isStreamFrozen(cameraId) {
    const detectionState = this.basicModeDetectionState.get(cameraId);
    return detectionState?.isFrozen || false;
  }

  // Check if detection is in progress
  isDetectionInProgress(cameraId) {
    const detectionState = this.basicModeDetectionState.get(cameraId);
    return detectionState?.detectionInProgress || false;
  }

  // Get last detection result
  getLastDetectionResult(cameraId) {
    const detectionState = this.basicModeDetectionState.get(cameraId);
    return detectionState?.lastDetectionResult || null;
  }

  // Get frozen frame (base64 image with detections)
  getFrozenFrame(cameraId) {
    const detectionState = this.basicModeDetectionState.get(cameraId);
    return detectionState?.frozenFrame || null;
  }

  // ===================
  // INITIALIZATION AND HEALTH CHECK UPDATES
  // ===================

  async initializeProcessor() {
    if (!this.canInitialize()) {
      throw new Error(`Cannot initialize from state: ${this.state}`);
    }

    console.log('🚀 Starting detection processor initialization...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.INITIALIZATION_TIMEOUT);

      // Check which mode to initialize based on system profile
      if (!this.systemProfile) {
        await this.updateSystemProfile();
      }

      let initEndpoint = '/api/detection/redis/initialize';
      if (this.shouldUseBasicMode()) {
        initEndpoint = '/api/detection/redis-basic/initialize'; // Updated to use Redis basic endpoint
        console.log('🔧 Initializing Redis basic detection mode');
      } else {
        console.log('🔧 Initializing optimized detection mode');
      }

      const response = await api.post(initEndpoint, {}, {
        signal: controller.signal,
        timeout: this.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running' || response.data.success) {
        console.log('✅ Detection processor initialized:', response.data.message);
        
        const modelResult = await this.loadModel(true);
        if (modelResult.success) {
          this.isModelLoaded = true;
          this.setState(DetectionStates.READY, 'Initialization completed');
          
          return {
            success: true,
            message: 'Detection system initialized and ready',
            state: this.state,
            mode: this.currentStreamingType
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

      // Choose health endpoint based on mode
      let healthEndpoint = '/api/detection/redis/health';
      if (this.shouldUseBasicMode()) {
        healthEndpoint = '/api/detection/redis-basic/health'; // Updated to use Redis basic endpoint
      }

      const response = await fetch(healthEndpoint, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      this.lastHealthCheck = Date.now();
      
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
      if (!this.isOperational() && !isInitialCheck && !isPostShutdownCheck) {
        throw new Error(`Cannot perform health check in state: ${this.state}`);
      }

      // Choose endpoints based on current mode
      let streamingHealthUrl = '/api/video_streaming/video/optimized/health';
      let detectionHealthUrl = '/api/detection/redis/health';
      
      if (this.shouldUseBasicMode()) {
        streamingHealthUrl = '/api/video_streaming/video/redis-basic/health'; // Updated for Redis basic
        detectionHealthUrl = '/api/detection/redis-basic/health'; // Updated for Redis basic
      }

      const healthCheckPromises = [
        Promise.race([
          api.get(streamingHealthUrl),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Streaming health check timeout')), 5000)
          )
        ]).catch(error => ({
          data: { status: 'unhealthy', error: error.message }
        })),
        
        Promise.race([
          api.get(detectionHealthUrl),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Detection health check timeout')), 5000)
          )
        ]).catch(error => ({
          data: { status: 'unhealthy', error: error.message }
        }))
      ];

      const [streamingHealth, detectionHealth] = await Promise.all(healthCheckPromises);
      this.lastHealthCheck = Date.now();

      if (isPostShutdownCheck) {
        this.hasPerformedPostShutdownCheck = true;
        console.log('✅ Post-shutdown health check completed and marked');
      }

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const detectionHealthy = detectionHealth.data.status === 'healthy';

      console.log(`🩺 Health check completed (${this.currentStreamingType} mode) - Streaming: ${streamingHealthy ? 'Healthy' : 'Unhealthy'}, Detection: ${detectionHealthy ? 'Healthy' : 'Unhealthy'}`);

      return {
        streaming: streamingHealth.data,
        detection: detectionHealth.data,
        overall: streamingHealthy && detectionHealthy,
        mode: this.currentStreamingType
      };
    } catch (error) {
      console.error("Error checking service health:", error);
      this.lastHealthCheck = Date.now();
      return {
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false,
        mode: this.currentStreamingType
      };
    } finally {
      this.healthCheckInProgress = false;
    }
  }

  async gracefulShutdown() {
    try {
      console.log('🛑 Initiating graceful detection shutdown...');
      this.setState(DetectionStates.SHUTTING_DOWN, 'Graceful shutdown requested');
      
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

      // Choose shutdown endpoint based on mode
      let shutdownEndpoint = '/api/detection/detection/shutdown/graceful';
      if (this.shouldUseBasicMode()) {
        // Redis basic mode cleanup
        console.log('ℹ️ Redis basic mode shutdown - stopping all streams and cleaning up');
        await this.stopAllStreams(false);
        
        try {
          await api.post('/api/detection/redis-basic/cleanup');
          await api.post('/api/video_streaming/video/redis-basic/cleanup');
        } catch (error) {
          console.warn('⚠️ Error during Redis basic cleanup:', error.message);
        }
        
        this.isModelLoaded = false;
        this.setState(DetectionStates.READY, 'Redis basic mode shutdown completed');
        return {
          success: true,
          message: 'Redis basic detection service shutdown completed'
        };
      }

      const response = await api.post(shutdownEndpoint, {}, {
        signal: controller.signal,
        timeout: this.SHUTDOWN_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      this.isModelLoaded = false;
      this.setState(DetectionStates.READY, 'Shutdown completed');
      
      console.log('✅ Graceful detection shutdown completed:', response.data);
      
      return {
        success: true,
        message: 'Detection service shutdown completed',
        details: response.data,
        mode: this.currentStreamingType
      };
      
    } catch (error) {
      console.error('❌ Error during graceful shutdown:', error);
      this.resetToInitializing('Shutdown failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Graceful shutdown timed out. Detection service may still be running.');
      }
      
      throw new Error(`Graceful shutdown failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getShutdownStatus() {
    try {
      let statusEndpoint = '/api/detection/detection/shutdown/status';
      if (this.shouldUseBasicMode()) {
        // For Redis basic mode, return simple status based on current state
        return {
          status: this.state === DetectionStates.SHUTTING_DOWN ? 'shutting_down' : 'ready',
          mode: 'redis_basic'
        };
      }

      const response = await api.get(statusEndpoint);
      return { ...response.data, mode: this.currentStreamingType };
    } catch (error) {
      console.error('Error getting shutdown status:', error);
      throw new Error(`Failed to get shutdown status: ${error.response?.data?.detail || error.message}`);
    }
  }

  // Enhanced optimized detection methods with mode awareness (keeping existing functionality)
  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    const {
      detectionFps = 5.0,
      streamQuality = 85,
      priority = 1
    } = options;

    try {
      console.log(`🎯 Starting optimized detection feed for camera ${cameraId} with target: ${targetLabel}`);

      if (!this.canStart()) {
        throw new Error(`Cannot start detection in state: ${this.state}. Current state must be READY.`);
      }

      if (!this.isModelLoaded) {
        throw new Error('Detection model is not loaded');
      }

      // Check if system supports optimized mode
      if (this.shouldUseBasicMode()) {
        console.log('⚠️ System in basic mode, falling back to Redis basic detection');
        return await this.startBasicDetectionFeed(cameraId, targetLabel, options);
      }

      await this.ensureCameraStarted(cameraId);
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
        detectionFps: detectionFps,
        mode: 'optimized'
      });

      this.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel,
        streamKey,
        startTime: Date.now(),
        isActive: true,
        type: 'optimized_detection'
      });

      this.setState(DetectionStates.RUNNING, `Started optimized detection for camera ${cameraId}`);
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

      if (!this.canStart()) {
        throw new Error(`Cannot start stream in state: ${this.state}. Current state must be READY.`);
      }

      // Check if system supports optimized mode
      if (this.shouldUseBasicMode()) {
        console.log('⚠️ System in basic mode, falling back to Redis basic stream');
        return await this.startBasicStream(cameraId, options);
      }

      await this.ensureCameraStarted(cameraId);
      await this.stopOptimizedStream(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/optimized/stream/${cameraId}?${params}`;
      
      this.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `optimized_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'optimized_stream'
      });

      if (this.state === DetectionStates.READY) {
        this.setState(DetectionStates.RUNNING, `Started optimized stream for camera ${cameraId}`);
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
      const stream = this.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.includes('optimized')) {
        console.log(`⏹️ Stopping optimized detection feed for camera ${cameraId}`);
        
        this.stopStatsMonitoring(cameraId);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping optimized stream API for camera ${cameraId}:`, error.message);
        }
        
        this.currentStreams.delete(cameraId);
        if (stream.streamKey) {
          this.detectionStats.delete(stream.streamKey);
        }
        
        if (this.currentStreams.size === 0 && this.state === DetectionStates.RUNNING) {
          this.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        if (performShutdown && !this.shouldUseBasicMode()) {
          try {
            console.log('🔄 Performing graceful detection shutdown...');
            await this.gracefulShutdown();
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
      const stream = this.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.includes('optimized')) {
        console.log(`⏹️ Stopping optimized stream for camera ${cameraId}`);
        
        try {
          await api.post(`/api/video_streaming/video/optimized/stream/${cameraId}/stop`);
        } catch (error) {
          console.warn(`⚠️ Error stopping optimized stream API for camera ${cameraId}:`, error.message);
        }
        
        this.currentStreams.delete(cameraId);
        
        if (this.currentStreams.size === 0 && this.state === DetectionStates.RUNNING) {
          this.setState(DetectionStates.READY, 'All streams stopped');
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
      
      const stopPromises = Array.from(this.currentStreams.keys()).map(cameraId => {
        const stream = this.currentStreams.get(cameraId);
        if (stream.type && stream.type.includes('redis_basic')) {
          return this.stopRedisBasicStream(cameraId);
        } else if (stream.type && stream.type.includes('basic')) {
          return this.stopBasicStream(cameraId);
        } else {
          return this.stopOptimizedDetectionFeed(cameraId, false);
        }
      });
      
      await Promise.allSettled(stopPromises);
      
      // Stop all streams on endpoints
      try {
        if (this.shouldUseOptimizedMode()) {
          await api.post('/api/video_streaming/video/optimized/streams/stop_all');
        }
        if (this.shouldUseBasicMode()) {
          await api.post('/api/video_streaming/video/redis-basic/cleanup');
        }
        await api.post('/api/video_streaming/video/basic/streams/stop_all');
      } catch (error) {
        console.warn('⚠️ Error calling stop_all APIs:', error.message);
      }
      
      this.currentStreams.clear();
      this.detectionStats.clear();
      this.basicModeDetectionState.clear(); // Clear Redis basic mode state
      
      for (const cameraId of this.eventListeners.keys()) {
        this.stopStatsMonitoring(cameraId);
      }
      
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
      
      console.log("✅ Stopped all streams");
    } catch (error) {
      console.error("❌ Error stopping all streams:", error);
      throw error;
    }
  }

  async reloadModel() {
    try {
      let reloadEndpoint = "/api/detection/detection/model/reload";
      if (this.shouldUseBasicMode()) {
        // Redis basic mode reinitializes instead of reloading
        await this.initializeProcessor();
        return { message: "Redis basic detection processor reinitialized" };
      }

      const response = await api.post(reloadEndpoint);
      console.log("✅ Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("❌ Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  }

  // Stats monitoring methods (enhanced for Redis basic mode awareness)
  startStatsMonitoring = (cameraId, streamKey) => {
    const pollInterval = setInterval(async () => {
      if (this.state === DetectionStates.SHUTTING_DOWN) {
        console.log(`⏭️ Skipping stats update for camera ${cameraId} - system shutting down`);
        return;
      }

      try {
        let statsEndpoint = `/api/video_streaming/video/optimized/stats/${cameraId}`;
        if (this.shouldUseBasicMode()) {
          statsEndpoint = `/api/video_streaming/video/redis-basic/stats`;
        }

        const response = await api.get(statsEndpoint);
        
        let streamStats;
        if (this.shouldUseBasicMode()) {
          // Redis basic mode returns different stats format
          streamStats = response.data.stream_stats?.find(s => s.camera_id === cameraId);
          
          // For Redis basic mode, also include detection state
          const detectionState = this.basicModeDetectionState.get(cameraId);
          if (streamStats && detectionState) {
            streamStats.is_frozen = detectionState.isFrozen;
            streamStats.detection_in_progress = detectionState.detectionInProgress;
            streamStats.last_detection_result = detectionState.lastDetectionResult;
          }
        } else {
          streamStats = response.data.streams?.[0];
        }
        
        if (streamStats && this.currentStreams.has(cameraId)) {
          const currentStats = this.detectionStats.get(streamKey) || {};
          
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
            detectionInProgress: streamStats.detection_in_progress || false,
            mode: this.currentStreamingType
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
        avgProcessingTime: 0,
        mode: this.currentStreamingType,
        isFrozen: this.isStreamFrozen(cameraId),
        detectionInProgress: this.isDetectionInProgress(cameraId)
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
          memory_usage_mb: 0,
          mode: this.currentStreamingType
        };
      }

      let statsEndpoint = '/api/video_streaming/video/optimized/stats';
      if (this.shouldUseBasicMode()) {
        statsEndpoint = '/api/video_streaming/video/redis-basic/stats';
      }

      const response = await api.get(statsEndpoint);
      return { ...response.data, mode: this.currentStreamingType };
    } catch (error) { 
      console.error("❌ Error getting streaming stats:", error);
      throw error;
    }
  }

  getPerformanceComparison = async () => {
    try {
      // Get comparison from optimized endpoint if available
      if (this.shouldUseOptimizedMode()) {
        const response = await api.get('/api/video_streaming/video/optimized/performance/comparison');
        return response.data;
      } else {
        // Return basic performance metrics for Redis basic mode
        return {
          current_mode: 'redis_basic',
          basic_performance: await this.getAllStreamingStats(),
          optimized_performance: null,
          recommendation: 'Redis basic mode - suitable for current system specifications with on-demand detection'
        };
      }
    } catch (error) {
      console.error("❌ Error getting performance comparison:", error);
      throw error;
    }
  };

  // Enhanced status methods with Redis basic mode support
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
      hasPerformedPostShutdownCheck: this.hasPerformedPostShutdownCheck,
      // Enhanced status with system info
      systemProfile: this.systemProfile,
      currentPerformanceMode: this.currentPerformanceMode,
      currentStreamingType: this.currentStreamingType,
      autoModeEnabled: this.autoModeEnabled,
      systemCapabilities: this.systemCapabilities,
      // Redis basic mode specific status
      basicModeStreams: Array.from(this.basicModeDetectionState.keys()),
      frozenStreams: Array.from(this.basicModeDetectionState.entries())
        .filter(([_, state]) => state.isFrozen)
        .map(([cameraId, _]) => cameraId),
      detectionsInProgress: Array.from(this.basicModeDetectionState.entries())
        .filter(([_, state]) => state.detectionInProgress)
        .map(([cameraId, _]) => cameraId)
    };
  }

  // Legacy method compatibility with auto mode (keeping existing functionality)
  startDetectionFeed = async (cameraId, targetLabel, options = {}) => {
    return this.startDetectionFeedWithAutoMode(cameraId, targetLabel, options);
  };

  stopDetectionFeed = async (cameraId, performShutdown = true) => {
    const stream = this.currentStreams.get(cameraId);
    if (stream) {
      if (stream.type && stream.type.includes('redis_basic')) {
        return this.stopRedisBasicStream(cameraId);
      } else if (stream.type && stream.type.startsWith('basic')) {
        return this.stopBasicStream(cameraId);
      } else {
        return this.stopOptimizedDetectionFeed(cameraId, performShutdown);
      }
    }
  };

  stopVideoStream = async (cameraId, performShutdown = true) => {
    return this.stopDetectionFeed(cameraId, performShutdown);
  };

  cleanup = async () => { 
    try {
      console.log('🧹 Starting DetectionService cleanup...');
      await this.stopAllStreams(true);
      
      // Clear Redis basic mode state
      this.basicModeDetectionState.clear();
      this.onDemandDetectionCallbacks.clear();
      
      console.log('✅ DetectionService cleanup completed');
    } catch (error) {
      console.error("❌ Error during cleanup:", error);
    }
  };

  // Keep all existing system profiling methods unchanged
  async forceSystemProfileRefresh() {
    try {
      console.log('🔄 Force refreshing system profile...');
      
      await api.post('/api/artifact_keeper/system/cache/refresh');
      await this.updateSystemProfile(true);
      
      return {
        success: true,
        profile: this.systemProfile,
        performanceMode: this.currentPerformanceMode,
        streamingType: this.currentStreamingType
      };
    } catch (error) {
      console.error('❌ Error force refreshing system profile:', error);
      throw error;
    }
  }

  async runPerformanceTest(durationSeconds = 10) {
    try {
      console.log(`🧪 Running performance test for ${durationSeconds} seconds...`);
      
      const response = await api.post('/api/artifact_keeper/system/performance/test', null, {
        params: { duration_seconds: durationSeconds }
      });
      
      await this.updateSystemProfile(true);
      return response.data;
    } catch (error) {
      console.error('❌ Error running performance test:', error);
      throw error;
    }
  }

  // Keep all existing system profile getters and methods
  getSystemProfile() {
    return this.systemProfile;
  }

  getCurrentPerformanceMode() {
    return this.currentPerformanceMode;
  }

  getCurrentStreamingType() {
    return this.currentStreamingType;
  }

  getSystemCapabilities() {
    return this.systemCapabilities;
  }

  shouldUseBasicMode() {
    return this.currentStreamingType === StreamingTypes.BASIC || 
           this.currentPerformanceMode === PerformanceModes.BASIC;
  }

  shouldUseOptimizedMode() {
    return this.currentStreamingType === StreamingTypes.OPTIMIZED;
  }

  // Keep all existing manual mode switching methods
  async switchToBasicMode() {
    try {
      console.log('🔄 Manually switching to Redis basic mode...');
      
      await this.stopAllStreams(false);
      
      this.currentPerformanceMode = PerformanceModes.BASIC;
      this.currentStreamingType = StreamingTypes.BASIC;
      this.autoModeEnabled = false;
      
      this.notifyProfileUpdateListeners();
      
      console.log('✅ Switched to Redis basic mode');
      return { success: true, mode: 'redis_basic' };
    } catch (error) {
      console.error('❌ Error switching to Redis basic mode:', error);
      throw error;
    }
  }
async switchToOptimizedMode() {
    try {
      console.log('🔄 Manually switching to optimized mode...');
      
      if (!this.systemProfile || this.systemProfile.performance_score < 30) {
        throw new Error('System does not meet minimum requirements for optimized mode');
      }
      
      await this.stopAllStreams(false);
      
      this.currentStreamingType = StreamingTypes.OPTIMIZED;
      if (this.currentPerformanceMode === PerformanceModes.BASIC) {
        this.currentPerformanceMode = PerformanceModes.STANDARD;
      }
      this.autoModeEnabled = false;
      
      this.notifyProfileUpdateListeners();
      
      console.log('✅ Switched to optimized mode');
      return { success: true, mode: 'optimized' };
    } catch (error) {
      console.error('❌ Error switching to optimized mode:', error);
      throw error;
    }
  }

  async enableAutoMode() {
    try {
      console.log('🤖 Enabling automatic mode selection...');
      this.autoModeEnabled = true;
      
      // Re-evaluate optimal settings
      await this.updateSystemProfile(true);
      
      console.log(`✅ Auto mode enabled - Selected: ${this.currentStreamingType}`);
      return { success: true, mode: this.currentStreamingType };
    } catch (error) {
      console.error('❌ Error enabling auto mode:', error);
      throw error;
    }
  }

  // Profile update listeners
  addProfileUpdateListener(listener) {
    this.profileUpdateListeners.add(listener);
    return () => this.profileUpdateListeners.delete(listener);
  }

  notifyProfileUpdateListeners() {
    this.profileUpdateListeners.forEach(listener => {
      try {
        listener({
          profile: this.systemProfile,
          performanceMode: this.currentPerformanceMode,
          streamingType: this.currentStreamingType,
          capabilities: this.systemCapabilities
        });
      } catch (error) {
        console.error('Error in profile update listener:', error);
      }
    });
  }

  // Camera monitoring method for system profiling
  async monitorCameraPerformance(cameraId, durationSeconds = 5) {
    try {
      console.log(`📊 Monitoring camera ${cameraId} performance for ${durationSeconds} seconds...`);
      
      const response = await api.get(`/api/artifact_keeper/system/performance/monitor/${cameraId}`, {
        params: { duration_seconds: durationSeconds }
      });
      
      return response.data;
    } catch (error) {
      console.error(`❌ Error monitoring camera ${cameraId} performance:`, error);
      throw error;
    }
  }

  // System information getters for external components
  getOptimalSettings() {
    return {
      performanceMode: this.currentPerformanceMode,
      streamingType: this.currentStreamingType,
      autoModeEnabled: this.autoModeEnabled,
      systemScore: this.systemProfile?.performance_score || 0,
      recommendedMode: this.systemProfile?.recommended_mode || 'basic'
    };
  }

  getSystemSummary() {
    if (!this.systemProfile) {
      return null;
    }

    return {
      cpu_cores: this.systemProfile.cpu_cores,
      total_memory_gb: this.systemProfile.total_memory_gb,
      available_memory_gb: this.systemProfile.available_memory_gb,
      gpu_available: this.systemProfile.gpu_available,
      gpu_name: this.systemProfile.gpu_name,
      cuda_available: this.systemProfile.cuda_available,
      performance_score: this.systemProfile.performance_score,
      performance_tier: this.systemProfile.performance_tier,
      meets_minimum_requirements: this.systemProfile.meets_minimum_requirements,
      current_mode: this.currentStreamingType,
      auto_mode_enabled: this.autoModeEnabled
    };
  }

  // ===================
  // REDIS BASIC MODE UTILITY METHODS
  // ===================

  // Get all basic mode streams with their status
  getBasicModeStreams() {
    const basicStreams = new Map();
    
    for (const [cameraId, stream] of this.currentStreams.entries()) {
      if (stream.type && stream.type.includes('redis_basic')) {
        const detectionState = this.basicModeDetectionState.get(cameraId);
        basicStreams.set(cameraId, {
          stream: stream,
          detection: detectionState || {
            isFrozen: false,
            lastDetectionTime: null,
            detectionInProgress: false,
            frozenFrame: null
          }
        });
      }
    }
    
    return basicStreams;
  }

  // Get basic mode stats summary
  getBasicModeStatsSummary() {
    const basicStreams = this.getBasicModeStreams();
    const totalStreams = basicStreams.size;
    let frozenCount = 0;
    let detectionsInProgress = 0;
    let totalDetections = 0;

    for (const [cameraId, streamInfo] of basicStreams.entries()) {
      if (streamInfo.detection.isFrozen) frozenCount++;
      if (streamInfo.detection.detectionInProgress) detectionsInProgress++;
      
      // Count detections from stats
      const statsKey = `redis_basic_${cameraId}_${streamInfo.stream.targetLabel}`;
      const stats = this.detectionStats.get(statsKey);
      if (stats) {
        totalDetections += stats.detectionCount || 0;
      }
    }

    return {
      totalStreams,
      frozenStreams: frozenCount,
      liveStreams: totalStreams - frozenCount,
      detectionsInProgress,
      totalDetections,
      mode: 'redis_basic'
    };
  }

  // Batch detection for multiple cameras
  async performBatchDetection(detectionRequests, options = {}) {
    try {
      const { maxConcurrent = 3, timeout = 15000 } = options;
      
      if (!Array.isArray(detectionRequests) || detectionRequests.length === 0) {
        throw new Error('Detection requests array is required');
      }

      if (detectionRequests.length > maxConcurrent) {
        throw new Error(`Maximum ${maxConcurrent} concurrent detections allowed in Redis basic mode`);
      }

      console.log(`🎯 Performing batch detection on ${detectionRequests.length} cameras`);

      const results = [];
      const detectionPromises = detectionRequests.map(async (request, index) => {
        try {
          const { cameraId, targetLabel, quality = 85, autoUnfreeze = false, unfreezeDelay = 2.0 } = request;
          
          if (cameraId === undefined || !targetLabel) {
            return {
              index,
              cameraId,
              success: false,
              error: 'cameraId and targetLabel are required'
            };
          }

          const result = await this.performOnDemandDetection(cameraId, targetLabel, {
            quality,
            autoUnfreeze,
            unfreezeDelay
          });

          return {
            index,
            cameraId,
            success: true,
            data: result.data,
            autoUnfreeze
          };

        } catch (error) {
          return {
            index,
            cameraId: request.cameraId,
            success: false,
            error: error.message
          };
        }
      });

      const batchResults = await Promise.allSettled(detectionPromises);
      
      batchResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        } else {
          results.push({
            index,
            cameraId: detectionRequests[index]?.cameraId,
            success: false,
            error: result.reason?.message || 'Unknown error'
          });
        }
      });

      const successful = results.filter(r => r.success).length;
      const failed = results.filter(r => !r.success).length;

      console.log(`✅ Batch detection completed: ${successful} successful, ${failed} failed`);

      return {
        success: true,
        results,
        summary: {
          total: results.length,
          successful,
          failed
        }
      };

    } catch (error) {
      console.error('❌ Error performing batch detection:', error);
      throw error;
    }
  }

  // Unfreeze all frozen streams
  async unfreezeAllStreams() {
    try {
      console.log('🔥 Unfreezing all frozen Redis basic streams...');
      
      const basicStreams = this.getBasicModeStreams();
      const frozenStreams = [];
      
      for (const [cameraId, streamInfo] of basicStreams.entries()) {
        if (streamInfo.detection.isFrozen) {
          frozenStreams.push(cameraId);
        }
      }

      if (frozenStreams.length === 0) {
        return {
          success: true,
          message: 'No frozen streams found',
          unfrozen: []
        };
      }

      const unfreezePromises = frozenStreams.map(async (cameraId) => {
        try {
          await this.unfreezeBasicStream(cameraId);
          return { cameraId, success: true };
        } catch (error) {
          console.error(`Error unfreezing stream ${cameraId}:`, error);
          return { cameraId, success: false, error: error.message };
        }
      });

      const results = await Promise.allSettled(unfreezePromises);
      const unfrozen = results
        .filter(r => r.status === 'fulfilled' && r.value.success)
        .map(r => r.value.cameraId);

      console.log(`✅ Unfroze ${unfrozen.length} of ${frozenStreams.length} streams`);

      return {
        success: true,
        message: `Successfully unfroze ${unfrozen.length} streams`,
        unfrozen,
        total: frozenStreams.length
      };

    } catch (error) {
      console.error('❌ Error unfreezing all streams:', error);
      throw error;
    }
  }

  // Enhanced detection stats for Redis basic mode
  getEnhancedDetectionStats(cameraId) {
    const stream = this.currentStreams.get(cameraId);
    const detectionState = this.basicModeDetectionState.get(cameraId);
    const basicStats = this.getDetectionStats(cameraId);

    if (!stream || !stream.type.includes('redis_basic')) {
      return null;
    }

    return {
      ...basicStats,
      streamInfo: {
        url: stream.url,
        startTime: stream.startTime,
        streamQuality: stream.streamQuality,
        targetLabel: stream.targetLabel,
        type: stream.type
      },
      detectionState: {
        isFrozen: detectionState?.isFrozen || false,
        detectionInProgress: detectionState?.detectionInProgress || false,
        lastDetectionTime: detectionState?.lastDetectionTime,
        frozenFrame: detectionState?.frozenFrame,
        lastDetectionResult: detectionState?.lastDetectionResult
      },
      capabilities: {
        canDetect: !detectionState?.detectionInProgress,
        canUnfreeze: detectionState?.isFrozen,
        hasResults: !!detectionState?.lastDetectionResult,
        hasFrozenFrame: !!detectionState?.frozenFrame
      }
    };
  }

  // Health check specifically for Redis basic mode
  async checkRedisBasicHealth() {
    try {
      const streamHealthPromise = api.get('/api/video_streaming/video/redis-basic/health');
      const detectionHealthPromise = api.get('/api/detection/redis-basic/health');

      const [streamHealth, detectionHealth] = await Promise.allSettled([
        streamHealthPromise,
        detectionHealthPromise
      ]);

      const streamHealthy = streamHealth.status === 'fulfilled' && 
                           streamHealth.value.data.status === 'healthy';
      const detectionHealthy = detectionHealth.status === 'fulfilled' && 
                              detectionHealth.value.data.status === 'healthy';

      return {
        streaming: {
          healthy: streamHealthy,
          status: streamHealth.status === 'fulfilled' ? streamHealth.value.data : { error: streamHealth.reason?.message }
        },
        detection: {
          healthy: detectionHealthy,
          status: detectionHealth.status === 'fulfilled' ? detectionHealth.value.data : { error: detectionHealth.reason?.message }
        },
        overall: streamHealthy && detectionHealthy,
        mode: 'redis_basic'
      };

    } catch (error) {
      console.error('❌ Error checking Redis basic health:', error);
      return {
        streaming: { healthy: false, error: error.message },
        detection: { healthy: false, error: error.message },
        overall: false,
        mode: 'redis_basic'
      };
    }
  }


  // ===================
  // STATE MANAGEMENT METHODS
  // ===================

  getState() {
    return this.state;
  }

  setState(newState, reason = '') {
    const oldState = this.state;
    this.state = newState;
    
    if (reason) {
      console.log(`🔄 State change: ${oldState} → ${newState} (${reason})`);
    }
    
    // Notify state change listeners
    this.notifyStateChangeListeners(newState, oldState);
  }

  addStateChangeListener(listener) {
    this.stateChangeListeners.add(listener);
    return () => this.stateChangeListeners.delete(listener);
  }

  notifyStateChangeListeners(newState, oldState) {
    this.stateChangeListeners.forEach(listener => {
      try {
        listener(newState, oldState);
      } catch (error) {
        console.error('Error in state change listener:', error);
      }
    });
  }

  resetToInitializing(reason = '') {
    console.log(`🔄 Resetting to INITIALIZING state: ${reason}`);
    this.setState(DetectionStates.INITIALIZING, reason);
    this.isModelLoaded = false;
    this.initializationPromise = null;
    this.hasPerformedInitialHealthCheck = false;
    this.hasPerformedPostShutdownCheck = false;
  }

  canStart() {
    return this.state === DetectionStates.READY && this.isModelLoaded;
  }

  canInitialize() {
    return this.state === DetectionStates.INITIALIZING || 
           this.state === DetectionStates.READY;
  }

  isOperational() {
    return this.state === DetectionStates.READY || 
           this.state === DetectionStates.RUNNING;
  }

  // Enhanced state management methods
  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    if (isInitialCheck && !this.hasPerformedInitialHealthCheck) {
      console.log('🩺 Allowing initial health check during initialization');
      return false;
    }

    if (isPostShutdownCheck && !this.hasPerformedPostShutdownCheck) {
      console.log('🩺 Allowing post-shutdown health check');
      return false;
    }

    if (this.state === DetectionStates.SHUTTING_DOWN) {
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
    if (this.isOperational() && this.isModelLoaded) {
      return { success: true, message: 'Already initialized', state: this.state };
    }

    if (this.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏳ Waiting for shutdown to complete before initializing...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      this.resetToInitializing('Post-shutdown reset');
    }

    if (this.state !== DetectionStates.INITIALIZING) {
      this.resetToInitializing('Ensure initialization');
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
        this.resetToInitializing('Failed initialization cleanup');
        throw error;
      }
    }

    this.initializationPromise = this.initializeProcessor();
    return await this.initializationPromise;
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



}

export const detectionService = new DetectionService();