import api from "../../../../utils/UseAxios";
import { BasicStreamManager } from './BasicStreamManager';
import { OptimizedStreamManager } from './OptimizedStreamManager';

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class StreamManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
    
    // Initialize composed managers
    this.basicManager = new BasicStreamManager(detectionService);
    this.optimizedManager = new OptimizedStreamManager(detectionService);
  }

  // ===================
  // FREEZE/UNFREEZE METHODS (delegated to BasicStreamManager)
  // ===================

  async freezeStream(cameraId) {
    return this.basicManager.freezeStream(cameraId);
  }

  async unfreezeStream(cameraId) {
    return this.basicManager.unfreezeStream(cameraId);
  }

  async getStreamFreezeStatus(cameraId) {
    return this.basicManager.getStreamFreezeStatus(cameraId);
  }

  // ===================
  // ENHANCED DETECTION METHODS WITH DATABASE INTEGRATION
  // ===================

  async createDetectionLot(lotName, expectedPieceId, expectedPieceNumber) {
    return this.basicManager.createDetectionLot(lotName, expectedPieceId, expectedPieceNumber);
  }

  async getDetectionLot(lotId) {
    return this.basicManager.getDetectionLot(lotId);
  }

  async updateLotTargetMatchStatus(lotId, isTargetMatch) {
    return this.basicManager.updateLotTargetMatchStatus(lotId, isTargetMatch);
  }

  async getLotDetectionSessions(lotId) {
    return this.basicManager.getLotDetectionSessions(lotId);
  }

  async performDetectionWithLotTracking(cameraId, targetLabel, options = {}) {
    return this.basicManager.performDetectionWithLotTracking(cameraId, targetLabel, options);
  }

  async createLotAndDetect(cameraId, lotName, expectedPieceId, expectedPieceNumber, targetLabel, options = {}) {
    return this.basicManager.createLotAndDetect(cameraId, lotName, expectedPieceId, expectedPieceNumber, targetLabel, options);
  }

  async performDetectionWithAutoCorrection(cameraId, lotId, targetLabel, options = {}) {
    return this.basicManager.performDetectionWithAutoCorrection(cameraId, lotId, targetLabel, options);
  }

  async unfreezeStreamAfterDetection(cameraId) {
    return this.basicManager.unfreezeStreamAfterDetection(cameraId);
  }

  // ===================
  // DETECTION METHODS
  // ===================

  async performOnDemandDetection(cameraId, targetLabel, options = {}) {
    // On-demand detection is only available in basic mode
    return this.basicManager.performOnDemandDetection(cameraId, targetLabel, options);
  }

  async performBatchDetection(detections = []) {
    // Batch detection is only available in basic mode
    return this.basicManager.performBatchDetection(detections);
  }

  // ===================
  // AUTO MODE METHODS (Main orchestration layer)
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
            return await this.optimizedManager.startOptimizedDetectionFeed(cameraId, targetLabel, options);
          } catch (optimizedError) {
            console.log(`⚠️ Optimized mode failed: ${optimizedError.message}, falling back to basic`);
            // Force switch to basic mode
            await this.detectionService.switchToBasicMode();
            return await this.basicManager.startBasicDetectionFeed(cameraId, targetLabel, options);
          }
        } else {
          console.log('🎯 Detection service not ready, using basic detection');
          // Ensure we're in basic mode
          if (!this.detectionService.shouldUseBasicMode()) {
            await this.detectionService.switchToBasicMode();
          }
          return await this.basicManager.startBasicDetectionFeed(cameraId, targetLabel, options);
        }
      } else {
        console.log(`🎯 Using basic detection for camera ${cameraId} (Performance mode: ${this.detectionService.currentPerformanceMode})`);
        return await this.basicManager.startBasicDetectionFeed(cameraId, targetLabel, options);
      }
      
    } catch (error) {
      console.error('❌ Error starting detection feed with auto mode:', error);
      
      // Final fallback to basic mode
      console.log('⚠️ Final fallback to basic mode due to error');
      try {
        await this.detectionService.switchToBasicMode();
        return await this.basicManager.startBasicDetectionFeed(cameraId, targetLabel, options);
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
        return await this.basicManager.startBasicStream(cameraId, options);
      } else {
        console.log(`📺 Attempting optimized streaming for camera ${cameraId} (Performance mode: ${this.detectionService.currentPerformanceMode})`);
        try {
          return await this.optimizedManager.startOptimizedStream(cameraId, options);
        } catch (optimizedError) {
          console.log(`⚠️ Optimized streaming failed: ${optimizedError.message}, falling back to basic`);
          // Force switch to basic mode
          await this.detectionService.switchToBasicMode();
          return await this.basicManager.startBasicStream(cameraId, options);
        }
      }
    } catch (error) {
      console.error('❌ Error starting stream with auto mode:', error);
      
      // Fallback to basic mode on error
      if (!this.detectionService.shouldUseBasicMode()) {
        console.log('⚠️ Falling back to basic mode due to error');
        try {
          await this.detectionService.switchToBasicMode();
          return await this.basicManager.startBasicStream(cameraId, options);
        } catch (basicError) {
          throw new Error(`All streaming modes failed: ${basicError.message}`);
        }
      }
      
      throw error;
    }
  }

  // ===================
  // BASIC STREAMING METHODS (delegated to BasicStreamManager)
  // ===================

  async startBasicDetectionFeed(cameraId, targetLabel, options = {}) {
    return this.basicManager.startBasicDetectionFeed(cameraId, targetLabel, options);
  }

  async startBasicStream(cameraId, options = {}) {
    return this.basicManager.startBasicStream(cameraId, options);
  }

  async stopBasicStream(cameraId) {
    return this.basicManager.stopBasicStream(cameraId);
  }

  async startBasicDetectionMonitoring(cameraId, targetLabel) {
    return this.basicManager.startBasicDetectionMonitoring(cameraId, targetLabel);
  }

  async startBasicDetectionPolling(cameraId, targetLabel) {
    return this.basicManager.startBasicDetectionPolling(cameraId, targetLabel);
  }

  // ===================
  // OPTIMIZED STREAMING METHODS (delegated to OptimizedStreamManager)
  // ===================

  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    return this.optimizedManager.startOptimizedDetectionFeed(cameraId, targetLabel, options);
  }

  async startOptimizedStream(cameraId, options = {}) {
    return this.optimizedManager.startOptimizedStream(cameraId, options);
  }

  async stopOptimizedDetectionFeed(cameraId, performShutdown = false) {
    return this.optimizedManager.stopOptimizedDetectionFeed(cameraId, performShutdown);
  }

  async stopOptimizedStream(cameraId) {
    return this.optimizedManager.stopOptimizedStream(cameraId);
  }

  // ===================
  // UNIFIED STREAM MANAGEMENT METHODS
  // ===================

  async stopAllStreams(performShutdown = true) {
    try {
      console.log('🛑 Stopping all streams...');
      
      const stopPromises = Array.from(this.detectionService.currentStreams.keys()).map(cameraId => {
        const stream = this.detectionService.currentStreams.get(cameraId);
        if (stream.type && stream.type.startsWith('basic')) {
          return this.basicManager.stopBasicStream(cameraId);
        } else {
          return this.optimizedManager.stopOptimizedDetectionFeed(cameraId, false);
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
            await this.basicManager.unfreezeStream(frozenStream.cameraId);
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
    // Use the appropriate manager based on current mode
    if (this.detectionService.shouldUseBasicMode()) {
      return this.basicManager.ensureCameraStarted(cameraId);
    } else {
      return this.optimizedManager.ensureCameraStarted(cameraId);
    }
  }

  // ===================
  // STATS MONITORING METHODS (unified interface)
  // ===================

  startStatsMonitoring = (cameraId, streamKey) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager.startBasicDetectionMonitoring(cameraId, stream.targetLabel);
    } else {
      return this.optimizedManager.startStatsMonitoring(cameraId, streamKey);
    }
  };

  stopStatsMonitoring = (cameraId) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager.stopStatsMonitoring(cameraId);
    } else {
      return this.optimizedManager.stopStatsMonitoring(cameraId);
    }
  };

  addStatsListener = (cameraId, callback) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager.addStatsListener(cameraId, callback);
    } else {
      return this.optimizedManager.addStatsListener(cameraId, callback);
    }
  };

  removeStatsListener = (cameraId, callback) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager.removeStatsListener(cameraId, callback);
    } else {
      return this.optimizedManager.removeStatsListener(cameraId, callback);
    }
  };

  notifyStatsListeners = (cameraId, stats) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager.notifyStatsListeners(cameraId, stats);
    } else {
      return this.optimizedManager.notifyStatsListeners(cameraId, stats);
    }
  };

  determineDetectionStatus = (streamStats) => {
    // Both managers have the same implementation, use basic as default
    return this.basicManager.determineDetectionStatus || this.optimizedManager.determineDetectionStatus;
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

      // Delegate to appropriate manager based on current mode
      if (this.detectionService.shouldUseBasicMode()) {
        return this.basicManager.getAllBasicStreamingStats();
      } else {
        return this.optimizedManager.getAllOptimizedStreamingStats();
      }
      
    } catch (error) { 
      console.error("❌ Error getting streaming stats:", error);
      throw error;
    }
  }

  getPerformanceComparison = async () => {
    try {
      // Get comparison from optimized endpoint if available
      if (this.detectionService.shouldUseOptimizedMode()) {
        return this.optimizedManager.getOptimizedPerformanceComparison();
      } else {
        // Return basic performance metrics for basic mode
        const basicStats = await this.basicManager.getAllBasicStreamingStats();
        return {
          current_mode: 'basic',
          basic_performance: basicStats,
          optimized_performance: null,
          recommendation: 'Basic mode - suitable for current system specifications',
          freeze_capability: true,
          on_demand_detection: true,
          enhanced_detection: true,
          database_integration: true
        };
      }
    } catch (error) {
      console.error("❌ Error getting performance comparison:", error);
      throw error;
    }
  };

  // ===================
  // LEGACY METHOD COMPATIBILITY
  // ===================

  stopDetectionFeed = async (cameraId, performShutdown = true) => {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream) {
      if (stream.type && stream.type.startsWith('basic')) {
        return this.basicManager.stopBasicStream(cameraId);
      } else {
        return this.optimizedManager.stopOptimizedDetectionFeed(cameraId, performShutdown);
      }
    }
  };

  // Legacy aliases for backward compatibility
  startDetectionFeed = async (cameraId, targetLabel, options = {}) => {
    return this.startDetectionFeedWithAutoMode(cameraId, targetLabel, options);
  };

  stopVideoStream = async (cameraId, performShutdown = true) => {
    return this.stopDetectionFeed(cameraId, performShutdown);
  };

  // ===================
  // ENHANCED DETECTION SERVICE HEALTH METHODS
  // ===================

  async getEnhancedDetectionHealth() {
    try {
      console.log('🏥 Checking enhanced detection service health...');
      
      const response = await api.get('/api/detection/basic/health');
      
      return {
        success: true,
        status: response.data.status,
        isInitialized: response.data.is_initialized,
        device: response.data.device,
        statistics: response.data.statistics,
        message: response.data.message
      };
    } catch (error) {
      console.error('❌ Error checking enhanced detection health:', error);
      return {
        success: false,
        status: 'unhealthy',
        error: error.response?.data?.detail || error.message
      };
    }
  }

  async initializeEnhancedDetectionProcessor() {
    try {
      console.log('🚀 Initializing enhanced detection processor...');
      
      const response = await api.post('/api/detection/basic/initialize');
      
      if (response.data.success) {
        console.log('✅ Enhanced detection processor initialized successfully');
        return {
          success: true,
          device: response.data.device,
          isInitialized: response.data.is_initialized,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to initialize enhanced detection processor');
      }
    } catch (error) {
      console.error('❌ Error initializing enhanced detection processor:', error);
      throw new Error(`Failed to initialize processor: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getEnhancedDetectionStats() {
    try {
      console.log('📊 Getting enhanced detection statistics...');
      
      const response = await api.get('/api/detection/basic/stats');
      
      return {
        success: true,
        stats: response.data.stats,
        timestamp: response.data.timestamp,
        serviceType: response.data.service_type
      };
    } catch (error) {
      console.error('❌ Error getting enhanced detection stats:', error);
      throw new Error(`Failed to get enhanced stats: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // CLEANUP METHODS
  // ===================

  cleanup = async () => { 
    try {
      console.log('🧹 Starting StreamManager cleanup...');
      
      // Cleanup both managers
      await Promise.allSettled([
        this.basicManager.cleanup(),
        this.optimizedManager.cleanup()
      ]);
      
      // Final cleanup
      await this.stopAllStreams(true);
      
      console.log('✅ StreamManager cleanup completed');
    } catch (error) {
      console.error("❌ Error during StreamManager cleanup:", error);
    }
  };
  // ===================
  // MODE-SPECIFIC DELEGATION METHODS
  // ===================

  /**
   * Get the appropriate manager based on current streaming type
   * @returns {BasicStreamManager|OptimizedStreamManager}
   */
  getCurrentManager() {
    return this.detectionService.shouldUseBasicMode() ? this.basicManager : this.optimizedManager;
  }

  /**
   * Get the manager for a specific camera's stream
   * @param {string|number} cameraId 
   * @returns {BasicStreamManager|OptimizedStreamManager}
   */
  getManagerForCamera(cameraId) {
    const stream = this.detectionService.currentStreams.get(cameraId);
    if (stream && stream.type && stream.type.startsWith('basic')) {
      return this.basicManager;
    } else {
      return this.optimizedManager;
    }
  }

  /**
   * Execute a method on the appropriate manager based on current mode
   * @param {string} methodName 
   * @param {...any} args 
   * @returns {Promise<any>}
   */
  async executeOnCurrentManager(methodName, ...args) {
    const manager = this.getCurrentManager();
    if (typeof manager[methodName] === 'function') {
      return manager[methodName](...args);
    } else {
      throw new Error(`Method ${methodName} not found on current manager`);
    }
  }

  /**
   * Execute a method on the manager responsible for a specific camera
   * @param {string|number} cameraId 
   * @param {string} methodName 
   * @param {...any} args 
   * @returns {Promise<any>}
   */
  async executeOnCameraManager(cameraId, methodName, ...args) {
    const manager = this.getManagerForCamera(cameraId);
    if (typeof manager[methodName] === 'function') {
      return manager[methodName](...args);
    } else {
      throw new Error(`Method ${methodName} not found on camera manager`);
    }
  }

  // ===================
  // UTILITY METHODS
  // ===================

  /**
   * Get information about all active streams and their managers
   * @returns {Object}
   */
  getStreamManagerInfo() {
    const streams = Array.from(this.detectionService.currentStreams.entries()).map(([cameraId, stream]) => ({
      cameraId: parseInt(cameraId),
      type: stream.type,
      manager: stream.type && stream.type.startsWith('basic') ? 'basic' : 'optimized',
      targetLabel: stream.targetLabel,
      startTime: stream.startTime,
      isActive: stream.isActive
    }));

    return {
      totalStreams: streams.length,
      basicStreams: streams.filter(s => s.manager === 'basic').length,
      optimizedStreams: streams.filter(s => s.manager === 'optimized').length,
      currentMode: this.detectionService.currentStreamingType,
      streams: streams
    };
  }

  /**
   * Check if any streams are currently active
   * @returns {boolean}
   */
  hasActiveStreams() {
    return this.detectionService.currentStreams.size > 0;
  }

  /**
   * Get all active camera IDs
   * @returns {number[]}
   */
  getActiveCameraIds() {
    return Array.from(this.detectionService.currentStreams.keys()).map(id => parseInt(id));
  }

  /**
   * Check if a specific camera has an active stream
   * @param {string|number} cameraId 
   * @returns {boolean}
   */
  isCameraActive(cameraId) {
    return this.detectionService.currentStreams.has(String(cameraId));
  }

  /**
   * Get stream information for a specific camera
   * @param {string|number} cameraId 
   * @returns {Object|null}
   */
  getCameraStreamInfo(cameraId) {
    const stream = this.detectionService.currentStreams.get(String(cameraId));
    if (stream) {
      return {
        cameraId: parseInt(cameraId),
        url: stream.url,
        targetLabel: stream.targetLabel,
        streamKey: stream.streamKey,
        startTime: stream.startTime,
        isActive: stream.isActive,
        type: stream.type,
        manager: stream.type && stream.type.startsWith('basic') ? 'basic' : 'optimized',
        isFrozen: this.detectionService.isStreamFrozen(String(cameraId)),
        stats: this.getDetectionStats(String(cameraId))
      };
    }
    return null;
  }
}