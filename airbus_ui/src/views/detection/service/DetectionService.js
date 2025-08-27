import { SystemProfiler } from './SystemProfiler';
import { StreamManager } from './StreamManger/MainStreamManager';
import { StateManager } from './StateManager';

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
    
    // NEW: Lot context tracking
    this.currentLotId = null;
    this.currentPieceLabel = null;
    this.isInitializedForLot = false;
    this.lotSpecificModelLoaded = false;
    
    // System profiling and performance mode
    this.systemProfile = null;
    this.currentPerformanceMode = PerformanceModes.BASIC; // Default to basic
    this.currentStreamingType = StreamingTypes.BASIC; // Default to basic
    this.systemCapabilities = null;
    this.profileUpdateListeners = new Set();
    this.autoModeEnabled = true; // Automatically switch modes based on specs
    
    // Initialization tracking
    this.initializationPromise = null;
    this.hasPerformedInitialHealthCheck = false;
    this.hasPerformedPostShutdownCheck = false;
    
    // Stream freeze tracking for basic mode
    this.frozenStreams = new Map(); // camera_id -> freeze_info
    this.freezeListeners = new Set();
    
    // Timeouts
    this.INITIALIZATION_TIMEOUT = 30000;
    this.HEALTH_CHECK_TIMEOUT = 10000;
    this.CAMERA_START_TIMEOUT = 15000;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.HEALTH_CHECK_COOLDOWN = 5000;
    this.SYSTEM_PROFILE_CACHE_DURATION = 30000; // 30 seconds cache
    
    console.log('🔧 DetectionService initialized with state:', this.state);
    
    // Initialize composed services
    this.systemProfiler = new SystemProfiler(this);
    this.streamManager = new StreamManager(this);
    this.stateManager = new StateManager(this);
    
    // Initialize system profiling but DON'T initialize detection yet
    this.initializeSystemProfiling();
  }

  // ===================
  // NEW: LOT-BASED INITIALIZATION METHODS
  // ===================

  /**
   * Initialize the detection system for a specific lot and piece
   * This replaces the old auto-initialization approach
   */
  async initializeForLot(lotId, pieceLabel) {
    console.log(`🎯 Initializing detection system for lot ${lotId} with piece: ${pieceLabel}`);
    
    try {
      // Check if already initialized for this specific lot
      if (this.isInitializedForLot && 
          this.currentLotId === lotId && 
          this.currentPieceLabel === pieceLabel && 
          this.state === DetectionStates.READY) {
        console.log('✅ Already initialized for this lot and piece');
        return {
          success: true,
          message: `Already initialized for ${pieceLabel}`,
          lotId: lotId,
          pieceLabel: pieceLabel
        };
      }

      // If initialized for a different lot, shutdown first
      if (this.isInitializedForLot && 
          (this.currentLotId !== lotId || this.currentPieceLabel !== pieceLabel)) {
        console.log(`🔄 Switching from lot ${this.currentLotId}(${this.currentPieceLabel}) to lot ${lotId}(${pieceLabel})`);
        await this.shutdownLotSpecificInitialization();
      }

      // Set lot context BEFORE initialization
      this.currentLotId = lotId;
      this.currentPieceLabel = pieceLabel;

      // Ensure system profile is updated
      if (!this.systemProfile) {
        await this.updateSystemProfile();
      }

      // Use lot-aware initialization from StateManager
      const result = await this.stateManager.ensureInitializedForLot(lotId, pieceLabel);

      if (result.success) {
        this.isInitializedForLot = true;
        this.lotSpecificModelLoaded = true;
        
        console.log(`✅ Detection system initialized successfully for lot ${lotId} with piece: ${pieceLabel}`);
        
        return {
          success: true,
          message: `Detection system ready for ${pieceLabel}`,
          lotId: lotId,
          pieceLabel: pieceLabel,
          mode: this.currentStreamingType,
          state: this.state
        };
      } else {
        throw new Error(result.message || 'Failed to initialize for lot');
      }

    } catch (error) {
      console.error(`❌ Failed to initialize for lot ${lotId}:`, error);
      
      // Clear lot context on failure
      this.currentLotId = null;
      this.currentPieceLabel = null;
      this.isInitializedForLot = false;
      this.lotSpecificModelLoaded = false;
      
      throw new Error(`Failed to initialize for ${pieceLabel}: ${error.message}`);
    }
  }

  /**
   * Shutdown lot-specific initialization to switch to another lot
   */
  async shutdownLotSpecificInitialization() {
    console.log(`🛑 Shutting down lot-specific initialization for lot ${this.currentLotId}`);
    
    try {
      // Stop all current streams first
      if (this.currentStreams.size > 0) {
        await this.stopAllStreams(false);
      }

      // Perform graceful shutdown if needed
      if (this.state === DetectionStates.RUNNING || this.state === DetectionStates.READY) {
        await this.gracefulShutdown();
      }

      // Clear lot-specific state
      this.currentLotId = null;
      this.currentPieceLabel = null;
      this.isInitializedForLot = false;
      this.lotSpecificModelLoaded = false;

      // Reset to initializing state for next lot
      this.stateManager.resetToInitializing('Switching to new lot');

      console.log('✅ Lot-specific shutdown completed');
      return { success: true };

    } catch (error) {
      console.error('❌ Error during lot-specific shutdown:', error);
      
      // Force reset on error
      this.currentLotId = null;
      this.currentPieceLabel = null;
      this.isInitializedForLot = false;
      this.lotSpecificModelLoaded = false;
      this.stateManager.resetToInitializing('Force reset due to shutdown error');
      
      throw error;
    }
  }

  /**
   * Check if system is ready for a specific lot
   */
  isReadyForLot(lotId, pieceLabel) {
    return this.isInitializedForLot &&
           this.currentLotId === lotId &&
           this.currentPieceLabel === pieceLabel &&
           this.state === DetectionStates.READY &&
           this.lotSpecificModelLoaded;
  }

  /**
   * Get current lot context
   */
  getCurrentLotContext() {
    return {
      lotId: this.currentLotId,
      pieceLabel: this.currentPieceLabel,
      isInitialized: this.isInitializedForLot,
      modelLoaded: this.lotSpecificModelLoaded
    };
  }

  // ===================
  // UPDATED: Enhanced detection service readiness check
  // ===================
  async ensureDetectionServiceReady(lotId = null, pieceLabel = null) {
    try {
      console.log('🔧 Ensuring detection service is ready...');
      
      // If lot context is provided, ensure initialization for that specific lot
      if (lotId && pieceLabel) {
        if (!this.isReadyForLot(lotId, pieceLabel)) {
          console.log(`🎯 Initializing for lot ${lotId} with piece ${pieceLabel}...`);
          const initResult = await this.initializeForLot(lotId, pieceLabel);
          return {
            success: initResult.success,
            message: initResult.message,
            fallbackMode: initResult.success ? null : 'basic',
            lotContext: true
          };
        } else {
          console.log(`✅ Already ready for lot ${lotId} with piece ${pieceLabel}`);
          return { 
            success: true, 
            message: `Ready for ${pieceLabel}`,
            lotContext: true
          };
        }
      }

      // Legacy check without lot context (not recommended but supported)
      const currentState = this.getState();
      if (currentState !== DetectionStates.READY) {
        console.log(`⚠️ Service not in READY state (current: ${currentState})`);
        return { 
          success: false, 
          message: `Service in ${currentState} state, not ready`,
          fallbackMode: 'basic'
        };
      }

      // Different readiness checks based on mode
      if (this.shouldUseBasicMode()) {
        console.log('✅ Basic mode - service is ready');
        return { success: true, message: 'Basic mode service ready' };
      }

      // For optimized mode, perform more thorough checks
      console.log('🔧 Checking optimized mode readiness...');

      // Check if processor initialization was successful
      try {
        const initResponse = await fetch('/api/detection/redis/initialize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
          timeout: 8000
        });

        if (initResponse.ok) {
          const initData = await initResponse.json();
          
          if (initData.status === 'already_running' || initData.status === 'initialized') {
            console.log('✅ Optimized detection processor is running');
            
            // Try health check but don't fail on 503
            try {
              const healthResponse = await fetch('/api/detection/redis/health', { timeout: 5000 });
              if (healthResponse.ok) {
                const healthData = await healthResponse.json();
                console.log('✅ Health check also passed');
                return { success: true, message: 'Optimized service fully ready' };
              } else if (healthResponse.status === 503) {
                console.log('⚠️ Health check returned 503 but processor is initialized - considering ready');
                return { 
                  success: true, 
                  message: 'Optimized service ready (health check pending)',
                  warning: 'Health check returned 503 but processor is running'
                };
              }
            } catch (healthError) {
              console.log('⚠️ Health check failed but processor is initialized - considering ready');
              return { 
                success: true, 
                message: 'Optimized service ready (health check unavailable)',
                warning: 'Health check failed but processor is running'
              };
            }
            
            return { success: true, message: 'Optimized service ready' };
          } else {
            console.log(`⚠️ Processor initialization returned: ${initData.status}`);
            return { 
              success: false, 
              message: `Processor status: ${initData.status}`,
              fallbackMode: 'basic'
            };
          }
        } else {
          console.log(`⚠️ Processor initialization failed with status: ${initResponse.status}`);
          return { 
            success: false, 
            message: `Processor initialization failed: ${initResponse.status}`,
            fallbackMode: 'basic'
          };
        }

      } catch (processorError) {
        console.log(`⚠️ Processor check failed: ${processorError.message}`);
        
        // Try recovery by attempting initialization
        console.log('🔧 Attempting processor recovery...');
        try {
          const recoveryResult = await this.initializeProcessor();
          if (recoveryResult.success) {
            console.log('✅ Processor recovery successful');
            return { success: true, message: 'Service recovered successfully' };
          }
        } catch (recoveryError) {
          console.log(`⚠️ Recovery failed: ${recoveryError.message}`);
        }

        return { 
          success: false, 
          message: 'Processor check failed, basic mode recommended',
          fallbackMode: 'basic'
        };
      }

    } catch (error) {
      console.error('❌ Error ensuring detection service ready:', error);
      return { 
        success: false, 
        message: error.message,
        fallbackMode: 'basic'
      };
    }
  }

  // ===================
  // SYSTEM PROFILING METHODS (delegated to SystemProfiler)
  // ===================
  async initializeSystemProfiling() {
    return this.systemProfiler.initializeSystemProfiling();
  }

  async updateSystemProfile(forceRefresh = false) {
    return this.systemProfiler.updateSystemProfile(forceRefresh);
  }

  determineOptimalSettings(profile, recommendation, capabilities) {
    return this.systemProfiler.determineOptimalSettings(profile, recommendation, capabilities);
  }

  async forceSystemProfileRefresh() {
    return this.systemProfiler.forceSystemProfileRefresh();
  }

  async runPerformanceTest(durationSeconds = 10) {
    return this.systemProfiler.runPerformanceTest(durationSeconds);
  }

  async monitorCameraPerformance(cameraId, durationSeconds = 5) {
    return this.systemProfiler.monitorCameraPerformance(cameraId, durationSeconds);
  }

  // System profile getters
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
      auto_mode_enabled: this.autoModeEnabled,
      // NEW: Lot context info
      current_lot_id: this.currentLotId,
      current_piece_label: this.currentPieceLabel,
      is_initialized_for_lot: this.isInitializedForLot
    };
  }

  // Manual mode switching
  async switchToBasicMode() {
    return this.systemProfiler.switchToBasicMode();
  }

  async switchToOptimizedMode() {
    return this.systemProfiler.switchToOptimizedMode();
  }

  async enableAutoMode() {
    return this.systemProfiler.enableAutoMode();
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

  // ===================
  // STREAM MANAGEMENT METHODS (delegated to StreamManager)
  // ===================

  // Freeze/Unfreeze methods
  async freezeStream(cameraId) {
    return this.streamManager.freezeStream(cameraId);
  }

  async unfreezeStream(cameraId) {
    return this.streamManager.unfreezeStream(cameraId);
  }

  async getStreamFreezeStatus(cameraId) {
    return this.streamManager.getStreamFreezeStatus(cameraId);
  }

  isStreamFrozen(cameraId) {
    return this.frozenStreams.has(cameraId);
  }

  getFrozenStreams() {
    return Array.from(this.frozenStreams.values());
  }

  addFreezeListener(listener) {
    this.freezeListeners.add(listener);
    return () => this.freezeListeners.delete(listener);
  }

  notifyFreezeListeners(cameraId, status) {
    this.freezeListeners.forEach(listener => {
      try {
        listener({
          cameraId: parseInt(cameraId),
          status,
          timestamp: Date.now()
        });
      } catch (error) {
        console.error('Error in freeze listener:', error);
      }
    });
  }

  // Detection methods
  async performOnDemandDetection(cameraId, targetLabel, options = {}) {
    return this.streamManager.performOnDemandDetection(cameraId, targetLabel, options);
  }

  async performBatchDetection(detections = []) {
    return this.streamManager.performBatchDetection(detections);
  }

  // UPDATED: Stream methods with lot context awareness
  async startDetectionFeedWithAutoMode(cameraId, targetLabel, options = {}) {
    // If we have lot context, ensure it's passed to the stream manager
    const enhancedOptions = {
      ...options,
      lotId: this.currentLotId,
      pieceLabel: this.currentPieceLabel
    };
    
    return this.streamManager.startDetectionFeedWithAutoMode(cameraId, targetLabel, enhancedOptions);
  }

  async startStreamWithAutoMode(cameraId, options = {}) {
    // If we have lot context, ensure it's passed to the stream manager
    const enhancedOptions = {
      ...options,
      lotId: this.currentLotId,
      pieceLabel: this.currentPieceLabel
    };
    
    return this.streamManager.startStreamWithAutoMode(cameraId, enhancedOptions);
  }

  // Basic stream methods
  async startBasicDetectionFeed(cameraId, targetLabel, options = {}) {
    return this.streamManager.startBasicDetectionFeed(cameraId, targetLabel, options);
  }

  async startBasicStream(cameraId, options = {}) {
    return this.streamManager.startBasicStream(cameraId, options);
  }

  async stopBasicStream(cameraId) {
    return this.streamManager.stopBasicStream(cameraId);
  }

  async startBasicDetectionMonitoring(cameraId, targetLabel) {
    return this.streamManager.startBasicDetectionMonitoring(cameraId, targetLabel);
  }

  async startBasicDetectionPolling(cameraId, targetLabel) {
    return this.streamManager.startBasicDetectionPolling(cameraId, targetLabel);
  }

  // Optimized stream methods
  async startOptimizedDetectionFeed(cameraId, targetLabel, options = {}) {
    return this.streamManager.startOptimizedDetectionFeed(cameraId, targetLabel, options);
  }

  async startOptimizedStream(cameraId, options = {}) {
    return this.streamManager.startOptimizedStream(cameraId, options);
  }

  async stopOptimizedDetectionFeed(cameraId, performShutdown = false) {
    return this.streamManager.stopOptimizedDetectionFeed(cameraId, performShutdown);
  }

  async stopOptimizedStream(cameraId) {
    return this.streamManager.stopOptimizedStream(cameraId);
  }

  async stopAllStreams(performShutdown = true) {
    return this.streamManager.stopAllStreams(performShutdown);
  }

  async ensureCameraStarted(cameraId) {
    return this.streamManager.ensureCameraStarted(cameraId);
  }

  // Stats monitoring methods
  startStatsMonitoring = (cameraId, streamKey) => {
    return this.streamManager.startStatsMonitoring(cameraId, streamKey);
  };

  stopStatsMonitoring = (cameraId) => {
    return this.streamManager.stopStatsMonitoring(cameraId);
  };

  addStatsListener = (cameraId, callback) => {
    return this.streamManager.addStatsListener(cameraId, callback);
  };

  removeStatsListener = (cameraId, callback) => {
    return this.streamManager.removeStatsListener(cameraId, callback);
  };

  notifyStatsListeners = (cameraId, stats) => {
    return this.streamManager.notifyStatsListeners(cameraId, stats);
  };

  determineDetectionStatus = (streamStats) => {
    return this.streamManager.determineDetectionStatus(streamStats);
  };

  getDetectionStats = (cameraId) => {
    return this.streamManager.getDetectionStats(cameraId);
  };

  async getAllStreamingStats() {
    return this.streamManager.getAllStreamingStats();
  }

  getPerformanceComparison = async () => {
    return this.streamManager.getPerformanceComparison();
  };

  // Legacy method compatibility
  startDetectionFeed = async (cameraId, targetLabel, options = {}) => {
    return this.startDetectionFeedWithAutoMode(cameraId, targetLabel, options);
  };

  stopDetectionFeed = async (cameraId, performShutdown = true) => {
    return this.streamManager.stopDetectionFeed(cameraId, performShutdown);
  };

  stopVideoStream = async (cameraId, performShutdown = true) => {
    return this.stopDetectionFeed(cameraId, performShutdown);
  };

  cleanup = async () => {
    return this.streamManager.cleanup();
  };

  // ===================
  // STATE MANAGEMENT METHODS (delegated to StateManager)
  // ===================

  // UPDATED: Lot-aware initialization methods
  async loadModel(isInitialCheck = false) {
    return this.stateManager.loadModel(isInitialCheck);
  }

  async loadModelWithLotContext(lotId, pieceLabel, isInitialCheck = false) {
    return this.stateManager.loadModelWithLotContext(lotId, pieceLabel, isInitialCheck);
  }

  async initializeProcessor() {
    return this.stateManager.initializeProcessor();
  }

  async initializeProcessorWithLot(lotId, pieceLabel) {
    return this.stateManager.initializeProcessorWithLot(lotId, pieceLabel);
  }

  async checkOptimizedHealth(isInitialCheck = false, isPostShutdownCheck = false) {
    return this.stateManager.checkOptimizedHealth(isInitialCheck, isPostShutdownCheck);
  }

  async gracefulShutdown() {
    return this.stateManager.gracefulShutdown();
  }

  async getShutdownStatus() {
    return this.stateManager.getShutdownStatus();
  }

  // UPDATED: Lot-aware ensureInitialized methods
  async ensureInitialized() {
    return this.stateManager.ensureInitialized();
  }

  async ensureInitializedForLot(lotId, pieceLabel) {
    return this.stateManager.ensureInitializedForLot(lotId, pieceLabel);
  }

  // State management methods
  setState(newState, reason = '') {
    return this.stateManager.setState(newState, reason);
  }

  getState() {
    return this.state;
  }

  addStateChangeListener(listener) {
    this.stateChangeListeners.add(listener);
    return () => this.stateChangeListeners.delete(listener);
  }

  canInitialize() {
    return this.state === DetectionStates.INITIALIZING;
  }

  canStart() {
    const canStart = this.state === DetectionStates.READY;
    console.log(`🔍 DetectionService.canStart(): state=${this.state}, canStart=${canStart}`);
    return canStart;
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
    return this.stateManager.resetToInitializing(reason);
  }

  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    return this.stateManager.shouldSkipHealthCheck(isInitialCheck, isPostShutdownCheck);
  }

  // Enhanced status methods
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

  // UPDATED: Enhanced status with lot context
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
      // NEW: Lot context status
      currentLotId: this.currentLotId,
      currentPieceLabel: this.currentPieceLabel,
      isInitializedForLot: this.isInitializedForLot,
      lotSpecificModelLoaded: this.lotSpecificModelLoaded
    };
  }
}

export { DetectionStates, PerformanceModes, StreamingTypes };
export const detectionService = new DetectionService();