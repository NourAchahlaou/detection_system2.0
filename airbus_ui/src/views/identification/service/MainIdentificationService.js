
// ===================
// MainIdentificationService.js
// ===================

import { IdentificationStreamManager } from './IdentificationStreamManager';
import { IdentificationStateManager } from './IdentificationStateManager';

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

  async stopAllStreams(performShutdown = true) {
    return this.streamManager.stopAllIdentificationStreams(performShutdown);
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

  async gracefulShutdown() {
    return this.stateManager.gracefulShutdown();
  }

  async getShutdownStatus() {
    return this.stateManager.getShutdownStatus();
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