// ===================
// MainIdentificationService.js
// ===================

import { IdentificationStreamManager } from './IdentificationStreamManager';
import { IdentificationStateManager } from './IdentificationStateManager';
import { IdentificationShutdownManager } from './IdentificationShutdownManager';

// Define the 4 states clearly
const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

class IdentificationService {
  constructor() {
    this.currentStreams = new Map();
    this.identificationStats = new Map();
    this.eventListeners = new Map();
    
    // Clear state management with proper initialization
    this.state = IdentificationStates.INITIALIZING;
    this.isModelLoaded = false;
    this.lastHealthCheck = null;
    this.healthCheckInProgress = false;
    this.stateChangeListeners = new Set();
    
    // Initialization tracking
    this.initializationPromise = null;
    this.hasPerformedInitialHealthCheck = false;
    this.hasPerformedPostShutdownCheck = false;
    
    // Stream freeze tracking
    this.frozenStreams = new Map(); // camera_id -> freeze_info
    this.freezeListeners = new Set();
    
    // Timeouts
    this.INITIALIZATION_TIMEOUT = 30000;
    this.HEALTH_CHECK_TIMEOUT = 10000;
    this.CAMERA_START_TIMEOUT = 15000;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.HEALTH_CHECK_COOLDOWN = 5000;
    
    console.log('🔧 IdentificationService initialized with state:', this.state);
    
    // Initialize composed services
    this.streamManager = new IdentificationStreamManager(this);
    this.stateManager = new IdentificationStateManager(this);
    this.shutdownManager = new IdentificationShutdownManager(this);
  }

  // ===================
  // STREAM FREEZE MANAGEMENT
  // ===================

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
          timestamp: Date.now(),
          purpose: 'identification'
        });
      } catch (error) {
        console.error('Error in identification freeze listener:', error);
      }
    });
  }

  // ===================
  // IDENTIFICATION METHODS
  // ===================

  async performPieceIdentification(cameraId, options = {}) {
    return this.streamManager.performPieceIdentification(cameraId, options);
  }

  async performQuickAnalysis(cameraId, options = {}) {
    return this.streamManager.performQuickAnalysis(cameraId, options);
  }

  async getAvailablePieceTypes() {
    return this.streamManager.getAvailablePieceTypes();
  }

  async updateConfidenceThreshold(threshold) {
    return this.streamManager.updateConfidenceThreshold(threshold);
  }

  async getIdentificationSettings() {
    return this.streamManager.getIdentificationSettings();
  }

  async getIdentificationHistory() {
    return this.streamManager.getIdentificationHistory();
  }

  async getIdentificationStats() {
    return this.streamManager.getIdentificationStats();
  }

  // ===================
  // STREAM MANAGEMENT METHODS
  // ===================

  async startIdentificationStream(cameraId, options = {}) {
    return this.streamManager.startIdentificationStream(cameraId, options);
  }

  async stopIdentificationStream(cameraId) {
    return this.streamManager.stopIdentificationStream(cameraId);
  }

  /**
   * Enhanced stopAllStreams with proper infrastructure shutdown
   * Delegates to shutdown manager for comprehensive cleanup
   */
  async stopAllStreams(performShutdown = true) {
    try {
      console.log('🛑 Stopping all identification streams with enhanced infrastructure cleanup...');
      
      // Delegate to shutdown manager for comprehensive stream shutdown
      await this.shutdownManager.stopAllStreamsWithInfrastructure(performShutdown);
      
      console.log("✅ Enhanced stopAllStreams completed");
      return {
        success: true,
        message: 'All identification streams stopped with infrastructure cleanup',
        streamsAffected: this.currentStreams.size,
        infrastructureShutdown: performShutdown
      };
      
    } catch (error) {
      console.error("❌ Error in enhanced stopAllStreams:", error);
      throw error;
    }
  }

  /**
   * Legacy method maintained for backward compatibility
   * Uses the stream manager's original implementation
   */
  async stopAllIdentificationStreams(performShutdown = true) {
    return this.streamManager.stopAllIdentificationStreams(performShutdown);
  }

  /**
   * Stop individual stream with enhanced infrastructure cleanup
   * Delegates to shutdown manager for comprehensive cleanup
   */
  async stopStreamWithInfrastructure(cameraId) {
    try {
      console.log(`🛑 Stopping identification stream for camera ${cameraId} with infrastructure cleanup...`);
      
      // Delegate to shutdown manager for comprehensive individual stream shutdown
      await this.shutdownManager.stopIdentificationStreamWithInfrastructure(cameraId);
      
      console.log(`✅ Enhanced stream stop completed for camera ${cameraId}`);
      return {
        success: true,
        message: `Identification stream stopped with infrastructure cleanup for camera ${cameraId}`,
        cameraId: parseInt(cameraId)
      };
      
    } catch (error) {
      console.error(`❌ Error stopping stream with infrastructure for camera ${cameraId}:`, error);
      throw error;
    }
  }

  async ensureCameraStarted(cameraId) {
    return this.streamManager.ensureCameraStarted(cameraId);
  }

  // Stats monitoring methods
  startStatsMonitoring = (cameraId) => {
    return this.streamManager.startStatsMonitoring(cameraId);
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

  getIdentificationStatsForCamera = (cameraId) => {
    const stream = this.currentStreams.get(cameraId);
    if (stream?.streamKey) {
      return this.identificationStats.get(stream.streamKey) || {
        piecesIdentified: 0,
        uniqueLabels: 0,
        labelCounts: {},
        lastIdentificationTime: null,
        avgProcessingTime: 0,
        isFrozen: this.isStreamFrozen(cameraId),
        mode: 'identification'
      };
    }
    return null;
  };

  async getAllStreamingStats() {
    return this.streamManager.getAllIdentificationStreamingStats();
  }

  cleanup = async () => {
    return this.streamManager.cleanup();
  };

  // ===================
  // STATE MANAGEMENT METHODS
  // ===================

  // Initialization and health check methods
  async loadModel(isInitialCheck = false) {
    return this.stateManager.loadModel(isInitialCheck);
  }

  async initializeProcessor() {
    return this.stateManager.initializeProcessor();
  }

  async ensureInitialized() {
    return this.stateManager.ensureInitialized();
  }

  async checkIdentificationHealth(isInitialCheck = false, isPostShutdownCheck = false) {
    return this.stateManager.checkIdentificationHealth(isInitialCheck, isPostShutdownCheck);
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
    return this.state === IdentificationStates.INITIALIZING;
  }

  canStart() {
    const canStart = this.state === IdentificationStates.READY;
    console.log(`🔍 IdentificationService.canStart(): state=${this.state}, canStart=${canStart}`);
    return canStart;
  }

  canStop() {
    return this.state === IdentificationStates.RUNNING;
  }

  canShutdown() {
    return [IdentificationStates.READY, IdentificationStates.RUNNING].includes(this.state);
  }

  isOperational() {
    return [IdentificationStates.READY, IdentificationStates.RUNNING].includes(this.state);
  }

  resetToInitializing(reason = 'Manual reset') {
    return this.stateManager.resetToInitializing(reason);
  }

  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    return this.stateManager.shouldSkipHealthCheck(isInitialCheck, isPostShutdownCheck);
  }

  // Enhanced status methods
  isReady() {
    return this.state === IdentificationStates.READY && this.isModelLoaded;
  }

  isRunning() {
    return this.state === IdentificationStates.RUNNING;
  }

  isInitializing() {
    return this.state === IdentificationStates.INITIALIZING;
  }

  isShuttingDown() {
    return this.state === IdentificationStates.SHUTTING_DOWN;
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
      frozenStreams: this.frozenStreams.size,
      lastHealthCheck: this.lastHealthCheck,
      healthCheckInProgress: this.healthCheckInProgress,
      hasPerformedInitialHealthCheck: this.hasPerformedInitialHealthCheck,
      hasPerformedPostShutdownCheck: this.hasPerformedPostShutdownCheck,
      mode: 'identification'
    };
  }

  // ===================
  // SHUTDOWN MANAGEMENT METHODS (Delegated to ShutdownManager)
  // ===================

  /**
   * Perform complete graceful shutdown of detection and identification services
   */
  async performCompleteShutdown() {
    return this.shutdownManager.performCompleteShutdown();
  }

  /**
   * Perform identification-only shutdown (leaves detection running)
   */
  async performIdentificationOnlyShutdown() {
    return this.shutdownManager.performIdentificationOnlyShutdown();
  }

  /**
   * Get detailed shutdown status from backend and frontend
   */
  async getShutdownStatus() {
    return this.shutdownManager.getShutdownStatus();
  }

  /**
   * Check if system can be shut down safely
   */
  async canShutdownSafely() {
    return this.shutdownManager.canShutdownSafely();
  }

  /**
   * Emergency shutdown - force stop everything immediately
   */
  async emergencyShutdown() {
    return this.shutdownManager.emergencyShutdown();
  }

  /**
   * Graceful shutdown with proper cleanup (delegates to shutdown manager)
   */
  async gracefulShutdown() {
    return this.shutdownManager.gracefulShutdown();
  }

  /**
   * Monitor shutdown progress
   */
  async monitorShutdownProgress(onProgress = null) {
    return this.shutdownManager.monitorShutdownProgress(onProgress);
  }

  /**
   * Get estimated shutdown time based on current state
   */
  getEstimatedShutdownTime() {
    return this.shutdownManager.getEstimatedShutdownTime();
  }

  /**
   * Check if shutdown is currently in progress
   */
  isShutdownInProgress() {
    return this.shutdownManager.isShutdownInProgress();
  }

  /**
   * Validate shutdown prerequisites
   */
  validateShutdownPrerequisites() {
    return this.shutdownManager.validateShutdownPrerequisites();
  }

  /**
   * Get available shutdown options based on current state
   */
  getShutdownOptions() {
    return this.shutdownManager.getShutdownOptions();
  }

  /**
   * Execute shutdown based on option ID
   */
  async executeShutdown(optionId, withMonitoring = true) {
    return this.shutdownManager.executeShutdown(optionId, withMonitoring);
  }

  /**
   * Reset all local state (used by shutdown manager)
   */
  resetLocalState() {
    return this.shutdownManager.resetLocalState();
  }

  /**
   * Reset only identification-specific state (used by shutdown manager)
   */
  resetIdentificationState() {
    return this.shutdownManager.resetIdentificationState();
  }

  /**
   * Enhanced method: Stop all streams with infrastructure cleanup
   * This is the preferred method for comprehensive stream shutdown
   */
  async stopAllStreamsWithInfrastructure(performCompleteShutdown = true) {
    return this.shutdownManager.stopAllStreamsWithInfrastructure(performCompleteShutdown);
  }

  /**
   * Enhanced method: Stop individual stream with infrastructure cleanup
   * This is the preferred method for comprehensive individual stream shutdown
   */
  async stopIdentificationStreamWithInfrastructure(cameraId) {
    return this.shutdownManager.stopIdentificationStreamWithInfrastructure(cameraId);
  }

  // ===================
  // UTILITY METHODS
  // ===================

  /**
   * Get information about all active identification streams
   * @returns {Object}
   */
  getStreamInfo() {
    const streams = Array.from(this.currentStreams.entries()).map(([cameraId, stream]) => ({
      cameraId: parseInt(cameraId),
      type: stream.type,
      startTime: stream.startTime,
      isActive: stream.isActive,
      isFrozen: this.isStreamFrozen(cameraId),
      purpose: stream.purpose
    }));

    return {
      totalStreams: streams.length,
      identificationStreams: streams.filter(s => s.type === 'identification_stream').length,
      frozenStreams: this.frozenStreams.size,
      currentState: this.state,
      streams: streams
    };
  }

  /**
   * Check if any identification streams are currently active
   * @returns {boolean}
   */
  hasActiveStreams() {
    return this.currentStreams.size > 0;
  }

  /**
   * Get all active camera IDs for identification
   * @returns {number[]}
   */
  getActiveCameraIds() {
    return Array.from(this.currentStreams.keys()).map(id => parseInt(id));
  }

  /**
   * Check if a specific camera has an active identification stream
   * @param {string|number} cameraId 
   * @returns {boolean}
   */
  isCameraActive(cameraId) {
    return this.currentStreams.has(String(cameraId));
  }

  /**
   * Get identification stream information for a specific camera
   * @param {string|number} cameraId 
   * @returns {Object|null}
   */
  getCameraStreamInfo(cameraId) {
    const stream = this.currentStreams.get(String(cameraId));
    if (stream) {
      return {
        cameraId: parseInt(cameraId),
        url: stream.url,
        streamKey: stream.streamKey,
        startTime: stream.startTime,
        isActive: stream.isActive,
        type: stream.type,
        purpose: stream.purpose,
        isFrozen: this.isStreamFrozen(String(cameraId)),
        stats: this.getIdentificationStatsForCamera(String(cameraId))
      };
    }
    return null;
  }
}

export { IdentificationStates };
export const identificationService = new IdentificationService();