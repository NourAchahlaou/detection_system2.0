import api from "../../../utils/UseAxios";
const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

// ===================
// IdentificationStateManager.js
// ===================

export class IdentificationStateManager {
  constructor(identificationService) {
    this.identificationService = identificationService;
  }

  async initializeProcessor() {
    if (!this.identificationService.canInitialize()) {
      throw new Error(`Cannot initialize from state: ${this.identificationService.state}`);
    }

    console.log('🚀 Starting identification processor initialization...');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.identificationService.INITIALIZATION_TIMEOUT);

      // Since identification uses the same processor as basic detection, 
      // we initialize the basic detection processor
      const response = await api.post('/api/detection/basic/initialize', {}, {
        signal: controller.signal,
        timeout: this.identificationService.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running' || response.data.success) {
        console.log('✅ Identification processor initialized:', response.data.message);
        
        // Wait for processor to fully start up
        console.log('⏳ Waiting for identification processor to fully initialize...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Perform health check
        const modelResult = await this.loadModel(true);
        
        if (modelResult.success) {
          this.identificationService.isModelLoaded = true;
          this.setState(IdentificationStates.READY, 'Identification initialization completed');
          
          return {
            success: true,
            message: 'Identification system initialized and ready',
            state: this.identificationService.state
          };
        } else {
          throw new Error('Model loading failed');
        }
      } else {
        throw new Error(`Unexpected initialization status: ${response.data.status}`);
      }
    } catch (error) {
      console.error('❌ Error initializing identification processor:', error);
      this.resetToInitializing('Identification initialization failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Identification initialization timed out. Please check if the detection service is running.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to identification service. Please ensure the backend is running.');
      } else {
        throw new Error(`Failed to initialize identification processor: ${error.response?.data?.detail || error.message}`);
      }
    } finally {
      this.identificationService.initializationPromise = null;
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

      this.identificationService.healthCheckInProgress = true;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.identificationService.HEALTH_CHECK_TIMEOUT);

      // Use basic detection health endpoint since identification shares the same processor
      const healthEndpoint = '/api/detection/basic/health';

      const response = await fetch(healthEndpoint, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      this.identificationService.lastHealthCheck = Date.now();
      
      if (isInitialCheck) {
        this.identificationService.hasPerformedInitialHealthCheck = true;
        console.log('✅ Initial identification health check completed and marked');
      }
      
      if (!response.ok) {
        if (response.status === 503) {
          console.log('🔄 Identification health check failed, model needs reloading...');
          throw new Error('Identification service not ready');
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }

      const result = await response.json();
      const modelLoaded = result.status === 'healthy';
      
      return {
        success: modelLoaded,
        message: modelLoaded ? 'Identification model loaded successfully' : 'Identification model not ready'
      };
      
    } catch (error) {
      console.error('Error loading identification model:', error);
      
      if (error.name === 'AbortError') {
        throw new Error('Identification health check timed out. Please check if the service is responding.');
      }
      
      throw new Error(`Failed to load identification model: ${error.message}`);
    } finally {
      this.identificationService.healthCheckInProgress = false;
    }
  }

  async gracefulShutdown() {
    try {
      console.log('🛑 Initiating graceful identification shutdown...');
      this.setState(IdentificationStates.SHUTTING_DOWN, 'Graceful identification shutdown requested');
      
      if (!this.identificationService.isOperational() && this.identificationService.state !== IdentificationStates.SHUTTING_DOWN) {
        console.log('ℹ️ Identification service already shut down');
        this.setState(IdentificationStates.READY, 'Already shut down');
        return {
          success: true,
          message: 'Identification service was already shut down'
        };
      }
      
      // For identification mode, just stop streams and unfreeze
      console.log('ℹ️ Identification mode shutdown - stopping all streams and unfreezing');
      await this.identificationService.stopAllStreams(false);
      
// Unfreeze all frozen streams
      const frozenStreams = this.identificationService.getFrozenStreams();
      for (const frozenStream of frozenStreams) {
        try {
          await this.identificationService.unfreezeStream(frozenStream.cameraId);
        } catch (error) {
          console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId} during shutdown:`, error.message);
        }
      }
      
      this.identificationService.isModelLoaded = false;
      this.setState(IdentificationStates.READY, 'Identification shutdown completed');
      
      return {
        success: true,
        message: 'Identification service shutdown completed'
      };
      
    } catch (error) {
      console.error('❌ Error during graceful identification shutdown:', error);
      this.resetToInitializing('Identification shutdown failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Graceful identification shutdown timed out. Service may still be running.');
      }
      
      throw new Error(`Graceful identification shutdown failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getShutdownStatus() {
    try {
      // For identification mode, return simple status based on current state
      return {
        status: this.identificationService.state === IdentificationStates.SHUTTING_DOWN ? 'shutting_down' : 'ready',
        mode: 'identification',
        frozen_streams: this.identificationService.getFrozenStreams().length,
        active_streams: this.identificationService.currentStreams.size
      };
    } catch (error) {
      console.error('Error getting identification shutdown status:', error);
      throw new Error(`Failed to get identification shutdown status: ${error.response?.data?.detail || error.message}`);
    }
  }

  async ensureInitialized() {
    if (this.identificationService.isOperational() && this.identificationService.isModelLoaded) {
      return { success: true, message: 'Identification already initialized', state: this.identificationService.state };
    }

    if (this.identificationService.state === IdentificationStates.SHUTTING_DOWN) {
      console.log('⏳ Waiting for identification shutdown to complete before initializing...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      this.resetToInitializing('Post-shutdown reset');
    }

    if (this.identificationService.state !== IdentificationStates.INITIALIZING) {
      this.resetToInitializing('Ensure identification initialization');
    }

    if (this.identificationService.initializationPromise) {
      try {
        console.log('⏳ Waiting for existing identification initialization to complete...');
        return await Promise.race([
          this.identificationService.initializationPromise,
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Identification initialization timeout')), this.identificationService.INITIALIZATION_TIMEOUT)
          )
        ]);
      } catch (error) {
        console.error('❌ Existing identification initialization failed or timed out:', error.message);
        this.resetToInitializing('Failed identification initialization cleanup');
        throw error;
      }
    }

    this.identificationService.initializationPromise = this.initializeProcessor();
    return await this.identificationService.initializationPromise;
  }

  // State management methods
  setState(newState, reason = '') {
    const oldState = this.identificationService.state;
    this.identificationService.state = newState;
    console.log(`🔄 Identification State changed: ${oldState} → ${newState}${reason ? ` (${reason})` : ''}`);
    
    // Reset health check flags on state transitions
    if (newState === IdentificationStates.INITIALIZING) {
      this.identificationService.hasPerformedInitialHealthCheck = false;
      this.identificationService.hasPerformedPostShutdownCheck = false;
    } else if (newState === IdentificationStates.READY && oldState === IdentificationStates.SHUTTING_DOWN) {
      this.identificationService.hasPerformedPostShutdownCheck = false;
    }
    
    // Notify all listeners
    this.identificationService.stateChangeListeners.forEach(listener => {
      try {
        listener(newState, oldState);
      } catch (error) {
        console.error('Error in identification state change listener:', error);
      }
    });
  }

  resetToInitializing(reason = 'Manual reset') {
    this.setState(IdentificationStates.INITIALIZING, reason);
    this.identificationService.isModelLoaded = false;
    this.identificationService.initializationPromise = null;
    this.identificationService.lastHealthCheck = null;
    this.identificationService.healthCheckInProgress = false;
    this.identificationService.hasPerformedInitialHealthCheck = false;
    this.identificationService.hasPerformedPostShutdownCheck = false;
    console.log('🔄 Identification service reset to initializing state');
  }

  shouldSkipHealthCheck(isInitialCheck = false, isPostShutdownCheck = false) {
    if (isInitialCheck && !this.identificationService.hasPerformedInitialHealthCheck) {
      console.log('🩺 Allowing initial identification health check during initialization');
      return false;
    }

    if (isPostShutdownCheck && !this.identificationService.hasPerformedPostShutdownCheck) {
      console.log('🩺 Allowing post-shutdown identification health check');
      return false;
    }

    if (this.identificationService.state === IdentificationStates.SHUTTING_DOWN) {
      console.log('⏭️ Skipping identification health check - system is shutting down');
      return true;
    }

    if (this.identificationService.healthCheckInProgress) {
      console.log('⏭️ Skipping identification health check - already in progress');
      return true;
    }

    if (this.identificationService.lastHealthCheck && (Date.now() - this.identificationService.lastHealthCheck) < this.identificationService.HEALTH_CHECK_COOLDOWN) {
      console.log('⏭️ Skipping identification health check - too soon since last check');
      return true;
    }

    return false;
  }

  async checkIdentificationHealth(isInitialCheck = false, isPostShutdownCheck = false) {
    if (this.shouldSkipHealthCheck(isInitialCheck, isPostShutdownCheck)) {
      return {
        streaming: { status: 'skipped', message: 'Health check skipped' },
        identification: { status: 'skipped', message: 'Health check skipped' },
        overall: false
      };
    }

    this.identificationService.healthCheckInProgress = true;

    try {
      if (!this.identificationService.isOperational() && !isInitialCheck && !isPostShutdownCheck) {
        throw new Error(`Cannot perform identification health check in state: ${this.identificationService.state}`);
      }

      // Use basic endpoints for identification
      const streamingHealthUrl = '/api/video_streaming/video/basic/health';
      const identificationHealthUrl = '/api/detection/basic/health';

      const healthCheckPromises = [
        // Streaming health check
        Promise.race([
          api.get(streamingHealthUrl),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Streaming health check timeout after 8 seconds')), 8000)
          )
        ]).catch(error => {
          console.warn(`⚠️ Streaming health check failed: ${error.message}`);
          return {
            data: { 
              status: 'unhealthy', 
              error: error.message,
              service: 'streaming'
            }
          };
        }),
        
        // Identification health check (using basic detection health)
        Promise.race([
          api.get(identificationHealthUrl),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Identification health check timeout after 8 seconds')), 8000)
          )
        ]).catch(error => {
          console.warn(`⚠️ Identification health check failed: ${error.message}`);
          return {
            data: { 
              status: 'unhealthy', 
              error: error.message,
              service: 'identification'
            }
          };
        })
      ];

      const [streamingHealth, identificationHealth] = await Promise.all(healthCheckPromises);
      this.identificationService.lastHealthCheck = Date.now();

      if (isPostShutdownCheck) {
        this.identificationService.hasPerformedPostShutdownCheck = true;
        console.log('✅ Post-shutdown identification health check completed and marked');
      }

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      const identificationHealthy = identificationHealth.data.status === 'healthy';
      const overall = streamingHealthy && identificationHealthy;

      console.log(`🩺 Identification health check completed - Streaming: ${streamingHealthy ? 'Healthy' : 'Unhealthy'}, Identification: ${identificationHealthy ? 'Healthy' : 'Unhealthy'}`);

      return {
        streaming: streamingHealth.data,
        identification: identificationHealth.data,
        overall: overall,
        mode: 'identification'
      };
    } catch (error) {
      console.error("Error checking identification service health:", error);
      this.identificationService.lastHealthCheck = Date.now();
      return {
        streaming: { status: 'unhealthy', error: error.message },
        identification: { status: 'unhealthy', error: error.message },
        overall: false,
        mode: 'identification'
      };
    } finally {
      this.identificationService.healthCheckInProgress = false;
    }
  }
}
