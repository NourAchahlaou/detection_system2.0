
import { SystemProfiler } from './SystemProfiler';
import { StreamManager } from './StreamManager';
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
    
    // Initialize system profiling
    this.initializeSystemProfiling();
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
      auto_mode_enabled: this.autoModeEnabled
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

  // Stream methods with auto mode
  async startDetectionFeedWithAutoMode(cameraId, targetLabel, options = {}) {
    return this.streamManager.startDetectionFeedWithAutoMode(cameraId, targetLabel, options);
  }

  async startStreamWithAutoMode(cameraId, options = {}) {
    return this.streamManager.startStreamWithAutoMode(cameraId, options);
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

  // Initialization and health check methods
  async loadModel(isInitialCheck = false) {
    return this.stateManager.loadModel(isInitialCheck);
  }

  async initializeProcessor() {
    return this.stateManager.initializeProcessor();
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

  async reloadModel() {
    return this.stateManager.reloadModel();
  }

  async ensureInitialized() {
    return this.stateManager.ensureInitialized();
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
      systemCapabilities: this.systemCapabilities
    };
  }
}

export { DetectionStates, PerformanceModes, StreamingTypes };
export const detectionService = new DetectionService();