import api from "../../../utils/UseAxios";

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class StateManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  // NEW: Lot-based model loading with piece context
  async loadModelWithLotContext(lotId, pieceLabel, isInitialCheck = false) {
    try {
      if (this.shouldSkipHealthCheck(isInitialCheck, false)) {
        return {
          success: false,
          message: 'Health check skipped due to system state'
        };
      }

      console.log(`🎯 Loading model with lot context: lotId=${lotId}, pieceLabel=${pieceLabel}`);
      
      this.detectionService.healthCheckInProgress = true;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.HEALTH_CHECK_TIMEOUT);

      // Choose health endpoint based on mode
      let healthEndpoint = '/api/detection/redis/health';
      let initEndpoint = '/api/detection/redis/initialize';
      
      if (this.detectionService.shouldUseBasicMode()) {
        healthEndpoint = '/api/detection/basic/health';
        initEndpoint = '/api/detection/basic/initialize';
      }

      // CRITICAL: Include piece context in the request
      const requestBody = {
        lot_id: lotId,
        piece_label: pieceLabel,
        target_label: pieceLabel
      };

      console.log(`🔧 Initializing with context:`, requestBody);

      // First ensure the processor is initialized with piece context
      const initResponse = await api.post(initEndpoint, requestBody, {
        signal: controller.signal,
        timeout: this.detectionService.INITIALIZATION_TIMEOUT
      });

      if (initResponse.data.status === 'initialized' || 
          initResponse.data.status === 'already_running' || 
          initResponse.data.success) {
        console.log('✅ Processor initialized with piece context:', initResponse.data.message);
        
        // Wait for processor to fully start up
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Now check health
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
          message: modelLoaded ? 
            `Detection model loaded successfully for piece: ${pieceLabel}` : 
            'Detection model not ready',
          pieceLabel: pieceLabel,
          lotId: lotId
        };
      } else {
        throw new Error(`Initialization failed: ${initResponse.data.status}`);
      }
      
    } catch (error) {
      console.error('Error loading detection model with lot context:', error);
      
      if (error.name === 'AbortError') {
        throw new Error('Health check timed out. Please check if the detection service is responding.');
      }
      
      throw new Error(`Failed to load detection model for ${pieceLabel}: ${error.message}`);
    } finally {
      this.detectionService.healthCheckInProgress = false;
    }
  }

  // UPDATED: Legacy method now redirects to lot-based loading
  async loadModel(isInitialCheck = false) {
    console.warn('⚠️ Using legacy loadModel - lot context recommended');
    
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

      // FIXED: Use proper API call with better error handling
      try {
        const response = await api.get(healthEndpoint, {
          signal: controller.signal,
          timeout: 8000
        });
        
        clearTimeout(timeoutId);
        this.detectionService.lastHealthCheck = Date.now();
        
        if (isInitialCheck) {
          this.detectionService.hasPerformedInitialHealthCheck = true;
          console.log('✅ Initial health check completed and marked');
        }
        
        const result = response.data;
        const modelLoaded = result.status === 'healthy';
        
        return {
          success: modelLoaded,
          message: modelLoaded ? 'Detection model loaded successfully' : 'Detection model not ready',
          healthData: result
        };
        
      } catch (healthError) {
        console.warn('Health check failed:', healthError.message);
        
        // For basic mode, consider processor operational even if health check fails
        if (this.detectionService.shouldUseBasicMode() && healthError.response?.status === 200) {
          return {
            success: true,
            message: 'Detection model initialized (health check pending)',
            warning: 'Health check failed but processor may be operational'
          };
        }
        throw healthError;
      }
      
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

  // NEW: Lot-specific processor initialization
  async initializeProcessorWithLot(lotId, pieceLabel) {
    if (!this.detectionService.canInitialize()) {
      throw new Error(`Cannot initialize from state: ${this.detectionService.state}`);
    }

    console.log(`🚀 Starting detection processor initialization with lot context...`);
    console.log(`📋 Lot ID: ${lotId}, Piece Label: ${pieceLabel}`);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.INITIALIZATION_TIMEOUT);

      // Check which mode to initialize based on system profile
      if (!this.detectionService.systemProfile) {
        await this.detectionService.updateSystemProfile();
      }

      let initEndpoint = '/api/detection/redis/initialize';
      let currentMode = 'optimized';
      
      if (this.detectionService.shouldUseBasicMode()) {
        initEndpoint = '/api/detection/basic/initialize';
        currentMode = 'basic';
        console.log('🔧 Initializing basic detection mode with piece context');
      } else {
        console.log('🔧 Initializing optimized detection mode with piece context');
      }

      // CRITICAL: Include lot and piece context in initialization
      const requestBody = {
        lot_id: lotId,
        piece_label: pieceLabel,
        target_label: pieceLabel,
        initialize_model_for_piece: true
      };

      console.log(`🔧 Calling initialization endpoint: ${initEndpoint} with context:`, requestBody);
      
      const response = await api.post(initEndpoint, requestBody, {
        signal: controller.signal,
        timeout: this.detectionService.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || 
          response.data.status === 'already_running' || 
          response.data.success) {
        
        console.log('✅ Detection processor initialized with piece context:', response.data.message);
        
        // Wait for the processor to fully start up
        console.log('⏳ Waiting for processor to fully initialize...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // FIXED: Perform proper health check
        console.log('🩺 Performing initial health check...');
        const healthResult = await this.checkOptimizedHealth(true, false);
        
        if (healthResult.overall || (currentMode === 'basic' && healthResult.detection.status !== 'unhealthy')) {
          this.detectionService.isModelLoaded = true;
          this.setState(DetectionStates.READY, `Initialization completed for ${pieceLabel}`);
          
          return {
            success: true,
            message: `Detection system initialized and ready for piece: ${pieceLabel}`,
            state: this.detectionService.state,
            mode: this.detectionService.currentStreamingType,
            pieceLabel: pieceLabel,
            lotId: lotId,
            healthCheck: healthResult
          };
        } else {
          // For basic mode, if processor is initialized, mark as ready even with health warnings
          if (currentMode === 'basic' && 
              (response.data.status === 'initialized' || response.data.status === 'already_running')) {
            console.warn(`⚠️ Basic processor initialized for ${pieceLabel} but health check shows issues - marking as ready with warning`);
            this.detectionService.isModelLoaded = true;
            this.setState(DetectionStates.READY, `Initialization completed for ${pieceLabel} with health warning`);
            
            return {
              success: true,
              message: `Detection system initialized for ${pieceLabel} (health check shows issues)`,
              state: this.detectionService.state,
              mode: this.detectionService.currentStreamingType,
              pieceLabel: pieceLabel,
              lotId: lotId,
              warning: 'Health check indicates potential issues but system is operational',
              healthCheck: healthResult
            };
          } else {
            throw new Error(`Health check failed for piece: ${pieceLabel}. System may not be fully operational.`);
          }
        }
      } else {
        throw new Error(`Unexpected initialization status: ${response.data.status}`);
      }
    } catch (error) {
      console.error(`❌ Error initializing detection processor for ${pieceLabel}:`, error);
      this.resetToInitializing('Initialization failed');
      
      if (error.name === 'AbortError') {
        throw new Error('Initialization timed out. Please check if the detection service is running.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to detection service. Please ensure the backend is running.');
      } else {
        throw new Error(`Failed to initialize detection processor for ${pieceLabel}: ${error.response?.data?.detail || error.message}`);
      }
    } finally {
      this.detectionService.initializationPromise = null;
    }
  }

  // UPDATED: Legacy method now requires lot context or uses default
  async initializeProcessor() {
    console.warn('⚠️ Using legacy initializeProcessor - lot context recommended');
    
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
      let currentMode = 'optimized';
      
      if (this.detectionService.shouldUseBasicMode()) {
        initEndpoint = '/api/detection/basic/initialize';
        currentMode = 'basic';
        console.log('🔧 Initializing basic detection mode');
      } else {
        console.log('🔧 Initializing optimized detection mode');
      }

      console.log(`🔧 Calling initialization endpoint: ${initEndpoint}`);
      const response = await api.post(initEndpoint, {}, {
        signal: controller.signal,
        timeout: this.detectionService.INITIALIZATION_TIMEOUT
      });
      
      clearTimeout(timeoutId);
      
      if (response.data.status === 'initialized' || response.data.status === 'already_running' || response.data.success) {
        console.log('✅ Detection processor initialized:', response.data.message);
        
        // Wait for the processor to fully start up
        console.log('⏳ Waiting for processor to fully initialize...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Perform model loading check
        let modelResult;
        if (currentMode === 'optimized') {
          modelResult = await this.loadModelWithRetries(true, 3);
        } else {
          modelResult = await this.loadModel(true);
        }
        
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
          // For optimized mode, if health check fails but processor is initialized,
          // still mark as ready but note the health issue
          if (currentMode === 'optimized' && 
              (response.data.status === 'initialized' || response.data.status === 'already_running')) {
            console.warn('⚠️ Optimized processor initialized but health check failed - marking as ready with warning');
            this.detectionService.isModelLoaded = true;
            this.setState(DetectionStates.READY, 'Initialization completed with health warning');
            
            return {
              success: true,
              message: 'Detection system initialized (health check pending)',
              state: this.detectionService.state,
              mode: this.detectionService.currentStreamingType,
              warning: 'Health check failed but processor is running'
            };
          } else {
            throw new Error('Model loading failed');
          }
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

  // UPDATED: Add retry mechanism for optimized mode health checks with lot context
  async loadModelWithRetries(isInitialCheck = false, maxRetries = 3, lotId = null, pieceLabel = null) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`🩺 Health check attempt ${attempt}/${maxRetries} for optimized mode...`);
        
        let result;
        if (lotId && pieceLabel) {
          result = await this.loadModelWithLotContext(lotId, pieceLabel, isInitialCheck);
        } else {
          result = await this.loadModel(isInitialCheck);
        }
        
        if (result.success) {
          console.log(`✅ Health check succeeded on attempt ${attempt}`);
          return result;
        }
        
        lastError = new Error(result.message);
        
        if (attempt < maxRetries) {
          console.log(`⏳ Health check attempt ${attempt} failed, retrying in 2 seconds...`);
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
      } catch (error) {
        lastError = error;
        console.log(`❌ Health check attempt ${attempt} failed: ${error.message}`);
        
        if (attempt < maxRetries) {
          console.log(`⏳ Retrying health check in 2 seconds...`);
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }
    }
    
    console.warn(`⚠️ All ${maxRetries} health check attempts failed for optimized mode`);
    return { success: false, message: lastError?.message || 'Health check failed after retries' };
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

      // FIXED: Better health check implementation with proper timeout and error handling
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
        
        // Detection health check with special handling for optimized mode
        Promise.race([
          api.get(detectionHealthUrl),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Detection health check timeout after 8 seconds')), 8000)
          )
        ]).catch(error => {
          console.warn(`⚠️ Detection health check failed: ${error.message}`);
          
          // FIXED: For optimized mode, if it's a 503 but we know the processor should be running,
          // check if it might just be starting up
          if (error.response?.status === 503 && !this.detectionService.shouldUseBasicMode()) {
            console.log('🔄 Optimized detection returned 503 - processor may still be starting');
            return {
              data: { 
                status: 'starting', 
                error: 'Processor initializing',
                service: 'detection',
                code: 503
              }
            };
          }
          
          return {
            data: { 
              status: 'unhealthy', 
              error: error.message,
              service: 'detection',
              code: error.response?.status
            }
          };
        })
      ];

      const [streamingHealth, detectionHealth] = await Promise.all(healthCheckPromises);
      this.detectionService.lastHealthCheck = Date.now();

      if (isPostShutdownCheck) {
        this.detectionService.hasPerformedPostShutdownCheck = true;
        console.log('✅ Post-shutdown health check completed and marked');
      }

      const streamingHealthy = streamingHealth.data.status === 'healthy';
      
      // FIXED: More nuanced detection health evaluation
      let detectionHealthy = detectionHealth.data.status === 'healthy';
      
      // For optimized mode, consider 'starting' status as potentially healthy
      if (!this.detectionService.shouldUseBasicMode() && 
          detectionHealth.data.status === 'starting' && 
          detectionHealth.data.code === 503) {
        console.log('🔄 Optimized detection is starting - considering as potentially healthy');
        detectionHealthy = true; // Optimistically consider as healthy for now
        detectionHealth.data.status = 'healthy';
        detectionHealth.data.message = 'Processor starting up';
      }

      const overall = streamingHealthy && detectionHealthy;

      console.log(`🩺 Health check completed (${this.detectionService.currentStreamingType} mode) - Streaming: ${streamingHealthy ? 'Healthy' : 'Unhealthy'}, Detection: ${detectionHealthy ? 'Healthy' : 'Unhealthy'}`);

      // FIXED: For optimized mode, if detection health failed due to 503, provide more context
      if (!this.detectionService.shouldUseBasicMode() && !detectionHealthy && detectionHealth.data.code === 503) {
        console.log('ℹ️ Optimized detection service may need more time to initialize');
        detectionHealth.data.recommendation = 'Service may need additional time to initialize';
      }

      return {
        streaming: streamingHealth.data,
        detection: detectionHealth.data,
        overall: overall,
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

  // All other existing methods remain the same...
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

  // NEW: Lot-aware initialization check
  async ensureInitializedForLot(lotId, pieceLabel) {
    console.log(`🔧 Ensuring system is initialized for lot ${lotId} with piece ${pieceLabel}`);

    // If already running with the same context, no need to reinitialize
    if (this.detectionService.isOperational() && this.detectionService.isModelLoaded) {
      // TODO: Add check to see if current model matches the required piece
      console.log('✅ System already initialized and running');
      return { success: true, message: 'Already initialized', state: this.detectionService.state };
    }

    if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
      console.log('⏳ Waiting for shutdown to complete before initializing...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      this.resetToInitializing('Post-shutdown reset');
    }

    if (this.detectionService.state !== DetectionStates.INITIALIZING) {
      this.resetToInitializing('Ensure initialization for lot');
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

    // Use lot-aware initialization
    this.detectionService.initializationPromise = this.initializeProcessorWithLot(lotId, pieceLabel);
    return await this.detectionService.initializationPromise;
  }

  // UPDATED: Legacy method with warning
  async ensureInitialized() {
    console.warn('⚠️ Using legacy ensureInitialized - consider ensureInitializedForLot for better piece context');
    
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

  // State management methods (unchanged)
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