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

export class StateManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  async loadModel(isInitialCheck = false) {
    try {
      if (this.shouldSkipHealthCheck(isInitialCheck, false)) {
        return {
          success: false,
          message: 'Health check skipped due to system state'
        };
      }

      this.detectionService.healthCheckInProgress = true;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.HEALTH_CHECK_TIMEOUT);

      // Choose health endpoint based on mode
      let healthEndpoint = '/api/detection/redis/health';
      if (this.detectionService.shouldUseBasicMode()) {
        healthEndpoint = '/api/detection/basic/health';
      }

      const response = await fetch(healthEndpoint, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      this.detectionService.lastHealthCheck = Date.now();
      
      if (isInitialCheck) {
        this.detectionService.hasPerformedInitialHealthCheck = true;
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
      this.detectionService.healthCheckInProgress = false;
    }
  }

  async initializeProcessor() {
    if (!this.detectionService.canInitialize()) {
      throw new Error(`Cannot initialize from state: ${this.detectionService.state}`);
    }

    console.log('🚀 Starting detection processor initialization...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.INITIALIZATION_TIMEOUT);

      // Check which mode to initialize based on system profile
      if (!this.detectionService.systemProfile) {
        await this.detectionService.updateSystemProfile();
      }

      let initEndpoint = '/api/detection/redis/initialize';
      if (this.detectionService.shouldUseBasicMode()) {
        initEndpoint = '/api/detection/basic/initialize';
        console.log('🔧 Initializing basic detection mode');
      } else {
        console.log('🔧 Initializing optimized detection mode');
      }

      const response = await api.post(initEndpoint, {}, {
        signal: controller.signal,
        timeout: this.detectionService.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running' || response.data.success) {
        console.log('✅ Detection processor initialized:', response.data.message);
        
        const modelResult = await this.loadModel(true);
        if (modelResult.success) {
          this.detectionService.isModelLoaded = true;
          this.setState(DetectionStates.READY, 'Initialization completed');
          
          return {
            success: true,
            message: 'Detection system initialized and ready',
            state: this.detectionService.state,
            mode: this.detectionService.currentStreamingType
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
      this.detectionService.initializationPromise = null;
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

    this.detectionService.healthCheckInProgress = true;

    try {
      if (!this.detectionService.isOperational() && !isInitialCheck && !isPostShutdownCheck) {
        throw new Error(`Cannot perform health check in state: ${this.detectionService.state}`);
      }

      // Choose endpoints based on current mode
      let streamingHealthUrl = '/api/video_streaming/video/optimized/health';
      let detectionHealthUrl = '/api/detection/redis/health';
      
      if (this.detectionService.shouldUseBasicMode()) {
        streamingHealthUrl = '/api/video_streaming/video/basic/health';
        detectionHealthUrl = '/api/detection/basic/health';
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
      this.detectionService.lastHealthCheck = Date.now();

      if (isPostShutdownCheck) {
        this.detectionService.hasPerformedPostShutdownCheck = true;
        console.log('✅ Post-shutdown health check completed and marked');
      }

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const detectionHealthy = detectionHealth.data.status === 'healthy';

      console.log(`🩺 Health check completed (${this.detectionService.currentStreamingType} mode) - Streaming: ${streamingHealthy ? 'Healthy' : 'Unhealthy'}, Detection: ${detectionHealthy ? 'Healthy' : 'Unhealthy'}`);

      return {
        streaming: streamingHealth.data,
        detection: detectionHealth.data,
        overall: streamingHealthy && detectionHealthy,
        mode: this.detectionService.currentStreamingType
      };
    } catch (error) {
      console.error("Error checking service health:", error);
      this.detectionService.lastHealthCheck = Date.now();
      return {
        streaming: { status: 'unhealthy', error: error.message },
        detection: { status: 'unhealthy', error: error.message },
        overall: false,
        mode: this.detectionService.currentStreamingType
      };
    } finally {
      this.detectionService.healthCheckInProgress = false;
    }
  }

  async gracefulShutdown() {
    try {
      console.log('🛑 Initiating graceful detection shutdown...');
      this.setState(DetectionStates.SHUTTING_DOWN, 'Graceful shutdown requested');
      
      if (!this.detectionService.isOperational() && this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
        console.log('ℹ️ Detection service already shut down');
        this.setState(DetectionStates.READY, 'Already shut down');
        return {
          success: true,
          message: 'Detection service was already shut down'
        };
      }
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.SHUTDOWN_TIMEOUT);

      // Choose shutdown endpoint based on mode
      let shutdownEndpoint = '/api/detection/detection/shutdown/graceful';
      if (this.detectionService.shouldUseBasicMode()) {
        // Basic mode doesn't need graceful shutdown, just stop streams and unfreeze
        console.log('ℹ️ Basic mode shutdown - stopping all streams and unfreezing');
        await this.detectionService.stopAllStreams(false);
        
        // Unfreeze all frozen streams
        const frozenStreams = this.detectionService.getFrozenStreams();
        for (const frozenStream of frozenStreams) {
          try {
            await this.detectionService.unfreezeStream(frozenStream.cameraId);
          } catch (error) {
            console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId} during shutdown:`, error.message);
          }
        }
        
        this.detectionService.isModelLoaded = false;
        this.setState(DetectionStates.READY, 'Basic mode shutdown completed');
        return {
          success: true,
          message: 'Basic detection service shutdown completed'
        };
      }

      const response = await api.post(shutdownEndpoint, {}, {
        signal: controller.signal,
        timeout: this.detectionService.SHUTDOWN_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      this.detectionService.isModelLoaded = false;
      this.setState(DetectionStates.READY, 'Shutdown completed');
      
      console.log('✅ Graceful detection shutdown completed:', response.data);
      
      return {
        success: true,
        message: 'Detection service shutdown completed',
        details: response.data,
        mode: this.detectionService.currentStreamingType
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
      if (this.detectionService.shouldUseBasicMode()) {
        // For basic mode, return simple status based on current state
        return {
          status: this.detectionService.state === DetectionStates.SHUTTING_DOWN ? 'shutting_down' : 'ready',
          mode: 'basic',
          frozen_streams: this.detectionService.getFrozenStreams().length
        };
      }

      const response = await api.get(statusEndpoint);
      return { ...response.data, mode: this.detectionService.currentStreamingType };
    } catch (error) {
      console.error('Error getting shutdown status:', error);
      throw new Error(`Failed to get shutdown status: ${error.response?.data?.detail || error.message}`);
    }
  }

  async reloadModel() {
    try {
      let reloadEndpoint = "/api/detection/detection/model/reload";
      if (this.detectionService.shouldUseBasicMode()) {
        // Basic mode reinitializes instead of reloading
        await this.initializeProcessor();
        return { message: "Basic detection processor reinitialized" };
      }

      const response = await api.post(reloadEndpoint);
      console.log("✅ Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("❌ Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  }

  async ensureInitialized() {
    if (this.detectionService.isOperational() && this.detectionService.isModelLoaded) {
      return { success: true, message: 'Already initialized', state: this.detectionService.state };
    }

    if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏳ Waiting for shutdown to complete before initializing...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      this.resetToInitializing('Post-shutdown reset');
    }

    if (this.detectionService.state !== DetectionStates.INITIALIZING) {
      this.resetToInitializing('Ensure initialization');
    }

    if (this.detectionService.initializationPromise) {
      try {
        console.log('⏳ Waiting for existing initialization to complete...');
        return await Promise.race([
          this.detectionService.initializationPromise,
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Initialization timeout')), this.detectionService.INITIALIZATION_TIMEOUT)
          )
        ]);
      } catch (error) {
        console.error('❌ Existing initialization failed or timed out:', error.message);
        this.resetToInitializing('Failed initialization cleanup');
        throw error;
      }
    }

    this.detectionService.initializationPromise = this.initializeProcessor();
    return await this.detectionService.initializationPromise;
  }

  // State management methods
  setState(newState, reason = '') {
    const oldState = this.detectionService.state;
    this.detectionService.state = newState;
    console.log(`🔄 State changed: ${oldState} → ${newState}${reason ? ` (${reason})` : ''}`);
    
    // Reset health check flags on state transitions
    if (newState === DetectionStates.INITIALIZING) {
      this.detectionService.hasPerformedInitialHealthCheck = false;
      this.detectionService.hasPerformedPostShutdownCheck = false;
    } else if (newState === DetectionStates.READY && oldState === DetectionStates.SHUTTING_DOWN) {
      this.detectionService.hasPerformedPostShutdownCheck = false;
    }
    
    // Notify all listeners
    this.detectionService.stateChangeListeners.forEach(listener => {
      try {
        listener(newState, oldState);
      } catch (error) {
        console.error('Error in state change listener:', error);
      }
    });
  }

  resetToInitializing(reason = 'Manual reset') {
    this.setState(DetectionStates.INITIALIZING, reason);
    this.detectionService.isModelLoaded = false;
    this.detectionService.initializationPromise = null;
    this.detectionService.lastHealthCheck = null;
    this.detectionService.healthCheckInProgress = false;
    this.detectionService.hasPerformedInitialHealthCheck = false;
    this.detectionService.hasPerformedPostShutdownCheck = false;
    console.log('🔄 Detection service reset to initializing state');
  }

  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    if (isInitialCheck && !this.detectionService.hasPerformedInitialHealthCheck) {
      console.log('🩺 Allowing initial health check during initialization');
      return false;
    }

    if (isPostShutdownCheck && !this.detectionService.hasPerformedPostShutdownCheck) {
      console.log('🩺 Allowing post-shutdown health check');
      return false;
    }

    if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏭️ Skipping health check - system is shutting down');
      return true;
    }

    if (this.detectionService.healthCheckInProgress) {
      console.log('⏭️ Skipping health check - already in progress');
      return true;
    }

    if (this.detectionService.lastHealthCheck && (Date.now() - this.detectionService.lastHealthCheck) < this.detectionService.HEALTH_CHECK_COOLDOWN) {
      console.log('⏭️ Skipping health check - too soon since last check');
      return true;
    }

    return false;
  }
}