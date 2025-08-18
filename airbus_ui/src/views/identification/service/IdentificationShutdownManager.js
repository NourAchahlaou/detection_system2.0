// ===================
// IdentificationShutdownManager.js
// ===================

import api from "../../../utils/UseAxios";

const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class IdentificationShutdownManager {
  constructor(identificationService) {
    this.identificationService = identificationService;
    this.SHUTDOWN_TIMEOUT = 35000;
    this.STATUS_CHECK_TIMEOUT = 10000;
  }

  // ===================
  // GRACEFUL SHUTDOWN METHODS
  // ===================

  /**
   * Perform complete graceful shutdown of detection and identification services
   */
  async performCompleteShutdown() {
    try {
      console.log('🛑 Initiating complete system shutdown (detection + identification)...');
      this.identificationService.setState(IdentificationStates.SHUTTING_DOWN, 'Complete system shutdown requested');

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.SHUTDOWN_TIMEOUT);

      try {
        // Stop all local streams first with proper video stream and camera shutdown
        await this.stopAllStreamsWithInfrastructure(true);

        // Call backend graceful shutdown endpoint
        const response = await api.post('/api/detection/shutdown/graceful', {}, {
          signal: controller.signal,
          timeout: this.SHUTDOWN_TIMEOUT
        });

        clearTimeout(timeoutId);

        if (response.data.status === 'shutdown_complete' || response.data.status === 'shutdown_partial') {
          console.log('✅ Complete system shutdown completed:', response.data.message);
          
          // Reset local state
          this.resetLocalState();
          this.identificationService.setState(IdentificationStates.INITIALIZING, 'Complete shutdown completed');

          return {
            success: true,
            status: response.data.status,
            message: response.data.message,
            results: response.data.results,
            servicesShutdown: response.data.services_shutdown,
            timestamp: response.data.timestamp
          };
        } else {
          throw new Error(`Unexpected shutdown status: ${response.data.status}`);
        }

      } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
          throw new Error('Complete shutdown timed out. Some services may still be running.');
        }

        throw error;
      }

    } catch (error) {
      console.error('❌ Error during complete system shutdown:', error);
      
      // Force reset to initializing state
      this.resetLocalState();
      this.identificationService.setState(IdentificationStates.INITIALIZING, 'Shutdown failed - force reset');
      
      if (error.name === 'AbortError') {
        throw new Error('Complete system shutdown timed out. Please check system status.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to shutdown service. Backend may already be down.');
      } else {
        throw new Error(`Complete shutdown failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  }

  /**
   * Perform identification-only shutdown (leaves detection running)
   */
  async performIdentificationOnlyShutdown() {
    try {
      console.log('🛑 Initiating identification-only shutdown...');
      this.identificationService.setState(IdentificationStates.SHUTTING_DOWN, 'Identification-only shutdown requested');

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.SHUTDOWN_TIMEOUT);

      try {
        // Stop all local streams first with proper video stream and camera shutdown
        await this.stopAllStreamsWithInfrastructure(false);

        // Call backend identification-only shutdown endpoint
        const response = await api.post('/api/detection/detection/shutdown/identification-only', {}, {
          signal: controller.signal,
          timeout: this.SHUTDOWN_TIMEOUT
        });

        clearTimeout(timeoutId);

        if (response.data.status === 'identification_shutdown_complete' || 
            response.data.status === 'identification_already_stopped') {
          console.log('✅ Identification-only shutdown completed:', response.data.message);
          
          // Reset only identification-related state
          this.resetIdentificationState();
          this.identificationService.setState(IdentificationStates.READY, 'Identification shutdown completed');

          return {
            success: true,
            status: response.data.status,
            message: response.data.message,
            identificationsPerformed: response.data.identifications_performed,
            detectionService: response.data.detection_service,
            timestamp: response.data.timestamp
          };
        } else {
          throw new Error(`Unexpected shutdown status: ${response.data.status}`);
        }

      } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
          throw new Error('Identification shutdown timed out. Service may still be running.');
        }

        throw error;
      }

    } catch (error) {
      console.error('❌ Error during identification-only shutdown:', error);
      
      // Reset identification state
      this.resetIdentificationState();
      this.identificationService.setState(IdentificationStates.READY, 'Identification shutdown failed - reset');
      
      if (error.name === 'AbortError') {
        throw new Error('Identification shutdown timed out. Please check service status.');
      } else if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to shutdown service. Backend may be down.');
      } else {
        throw new Error(`Identification shutdown failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  }

  // ===================
  // ENHANCED STREAM SHUTDOWN WITH INFRASTRUCTURE
  // ===================

  /**
   * Stop all identification streams with proper video stream and camera shutdown
   * Similar to the detection service's stopAllStreams method
   */
  async stopAllStreamsWithInfrastructure(performCompleteShutdown = true) {
    try {
      console.log('🛑 Stopping all identification streams with infrastructure shutdown...');
      
      const stopPromises = Array.from(this.identificationService.currentStreams.keys()).map(cameraId => {
        return this.stopIdentificationStreamWithInfrastructure(cameraId);
      });
      
      await Promise.allSettled(stopPromises);
       
      // Unfreeze all frozen identification streams
      const frozenStreams = this.identificationService.getFrozenStreams();
      for (const frozenStream of frozenStreams) {
        try {
          await this.identificationService.unfreezeStream(frozenStream.cameraId);
        } catch (error) {
          console.warn(`⚠️ Error unfreezing identification stream ${frozenStream.cameraId}:`, error.message);
        }
      }
      
      // Stop cameras if this is a complete shutdown
      if (performCompleteShutdown) {
        try {
          console.log('📹 Stopping all cameras...');
          await api.post("/api/artifact_keeper/camera/stop");
          console.log('✅ All cameras stopped');
        } catch (error) {
          console.warn('⚠️ Error stopping cameras:', error.message);
        }
      }
      
      // Clear local state
      this.identificationService.currentStreams.clear();
      this.identificationService.identificationStats.clear();
      
      // Stop all stats monitoring
      for (const cameraId of this.identificationService.eventListeners.keys()) {
        this.identificationService.stopStatsMonitoring(cameraId);
      }
      
      if (this.identificationService.state === IdentificationStates.RUNNING) {
        this.identificationService.setState(IdentificationStates.READY, 'All identification streams stopped');
      }
      
      console.log("✅ Stopped all identification streams with infrastructure");
      
    } catch (error) {
      console.error("❌ Error stopping all identification streams with infrastructure:", error);
      throw error;
    }
  }

  /**
   * Stop individual identification stream with proper video stream and camera cleanup
   */
  async stopIdentificationStreamWithInfrastructure(cameraId) {
    try {
      const stream = this.identificationService.currentStreams.get(cameraId);
      if (stream) {
        console.log(`⏹️ Stopping identification stream for camera ${cameraId} with infrastructure cleanup`);
        
        // Unfreeze stream if it's frozen
        if (this.identificationService.isStreamFrozen(cameraId)) {
          try {
            await this.identificationService.unfreezeStream(cameraId);
          } catch (error) {
            console.warn(`⚠️ Error unfreezing identification stream during stop for camera ${cameraId}:`, error.message);
          }
        }
        
        // Stop stats monitoring
        this.identificationService.stopStatsMonitoring(cameraId);
        
        try {
          // Stop the video stream for this specific camera
          console.log(`📺 Stopping video stream for camera ${cameraId}...`);
          await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/stop`);
          console.log(`✅ Video stream stopped for camera ${cameraId}`);
          
        } catch (error) {
          console.warn(`⚠️ Error stopping video stream for camera ${cameraId}:`, error.message);
        }
        
        // Remove from current streams
        this.identificationService.currentStreams.delete(cameraId);
        
        // Update state based on remaining streams
        if (this.identificationService.currentStreams.size === 0 && 
            this.identificationService.state === IdentificationStates.RUNNING) {
          this.identificationService.setState(IdentificationStates.READY, 'All identification streams stopped');
        }
        
        console.log(`✅ Successfully stopped identification stream for camera ${cameraId} with infrastructure`);
      }
    } catch (error) {
      console.error(`❌ Error stopping identification stream with infrastructure for camera ${cameraId}:`, error);
      throw error;
    }
  }

  // ===================
  // SHUTDOWN STATUS METHODS
  // ===================

  /**
   * Get detailed shutdown status from backend
   */
  async getShutdownStatus() {
    try {
      console.log('📊 Getting shutdown status...');

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.STATUS_CHECK_TIMEOUT);

      try {
        const response = await api.get('/api/detection/detection/shutdown/status', {
          signal: controller.signal,
          timeout: this.STATUS_CHECK_TIMEOUT
        });

        clearTimeout(timeoutId);

        const statusData = response.data;
        
        // Enhance with local frontend state
        const enhancedStatus = {
          ...statusData,
          frontend: {
            identificationState: this.identificationService.state,
            isModelLoaded: this.identificationService.isModelLoaded,
            activeStreams: this.identificationService.currentStreams.size,
            frozenStreams: this.identificationService.frozenStreams.size,
            lastHealthCheck: this.identificationService.lastHealthCheck,
            healthCheckInProgress: this.identificationService.healthCheckInProgress
          }
        };

        console.log('✅ Shutdown status retrieved:', enhancedStatus.status);
        return enhancedStatus;

      } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
          throw new Error('Shutdown status check timed out.');
        }

        throw error;
      }

    } catch (error) {
      console.error('❌ Error getting shutdown status:', error);
      
      if (error.code === 'ECONNREFUSED') {
        return {
          status: 'backend_offline',
          message: 'Cannot connect to backend - may already be shut down',
          frontend: {
            identificationState: this.identificationService.state,
            isModelLoaded: this.identificationService.isModelLoaded,
            activeStreams: this.identificationService.currentStreams.size,
            frozenStreams: this.identificationService.frozenStreams.size
          }
        };
      }

      throw new Error(`Failed to get shutdown status: ${error.response?.data?.detail || error.message}`);
    }
  }

  /**
   * Check if system can be shut down safely
   */
  async canShutdownSafely() {
    try {
      const status = await this.getShutdownStatus();
      
      // Check frontend state
      const frontendCanShutdown = this.identificationService.canShutdown();
      
      // Check if there are active processes that might be disrupted
      const hasActiveStreams = this.identificationService.currentStreams.size > 0;
      const hasFrozenStreams = this.identificationService.frozenStreams.size > 0;
      
      const canShutdown = status.can_shutdown && frontendCanShutdown;
      
      return {
        canShutdown,
        reasons: {
          backendStatus: status.can_shutdown,
          frontendState: frontendCanShutdown,
          activeStreams: hasActiveStreams,
          frozenStreams: hasFrozenStreams
        },
        estimatedShutdownTime: status.estimated_shutdown_time_seconds || 30,
        warnings: this.getShutdownWarnings(hasActiveStreams, hasFrozenStreams)
      };

    } catch (error) {
      console.error('❌ Error checking shutdown safety:', error);
      return {
        canShutdown: false,
        error: error.message,
        estimatedShutdownTime: 30
      };
    }
  }

  /**
   * Get warnings about potential issues during shutdown
   */
  getShutdownWarnings(hasActiveStreams, hasFrozenStreams) {
    const warnings = [];
    
    if (hasActiveStreams) {
      warnings.push(`${this.identificationService.currentStreams.size} active identification streams will be stopped`);
      warnings.push('Video streams will be terminated');
      warnings.push('Camera feeds will be stopped');
    }
    
    if (hasFrozenStreams) {
      warnings.push(`${this.identificationService.frozenStreams.size} frozen identification streams will be unfrozen`);
    }
    
    if (this.identificationService.healthCheckInProgress) {
      warnings.push('Health check in progress - may cause brief delay');
    }
    
    return warnings;
  }

  // ===================
  // STATE RESET METHODS
  // ===================

  /**
   * Reset all local state (complete reset)
   */
  resetLocalState() {
    console.log('🔄 Resetting all identification service state...');
    
    // Clear streams and stats
    this.identificationService.currentStreams.clear();
    this.identificationService.identificationStats.clear();
    this.identificationService.frozenStreams.clear();
    
    // Stop all monitoring intervals
    for (const [cameraId, eventData] of this.identificationService.eventListeners.entries()) {
      if (eventData.monitorInterval) {
        clearInterval(eventData.monitorInterval);
      }
      if (eventData.pollInterval) {
        clearInterval(eventData.pollInterval);
      }
    }
    this.identificationService.eventListeners.clear();
    
    // Reset flags and promises
    this.identificationService.isModelLoaded = false;
    this.identificationService.initializationPromise = null;
    this.identificationService.lastHealthCheck = null;
    this.identificationService.healthCheckInProgress = false;
    this.identificationService.hasPerformedInitialHealthCheck = false;
    this.identificationService.hasPerformedPostShutdownCheck = false;
    
    console.log('✅ All identification service state reset');
  }

  /**
   * Reset only identification-specific state (keeps detection running)
   */
  resetIdentificationState() {
    console.log('🔄 Resetting identification-specific state...');
    
    // Clear identification streams and stats
    this.identificationService.identificationStats.clear();
    this.identificationService.frozenStreams.clear();
    
    // Stop monitoring for identification streams only
    for (const [cameraId, stream] of this.identificationService.currentStreams.entries()) {
      if (stream.type === 'identification_stream') {
        const eventData = this.identificationService.eventListeners.get(cameraId);
        if (eventData?.monitorInterval) {
          clearInterval(eventData.monitorInterval);
        }
        if (eventData?.pollInterval) {
          clearInterval(eventData.pollInterval);
        }
        this.identificationService.eventListeners.delete(cameraId);
      }
    }
    
    // Remove identification streams
    for (const [cameraId, stream] of this.identificationService.currentStreams.entries()) {
      if (stream.type === 'identification_stream') {
        this.identificationService.currentStreams.delete(cameraId);
      }
    }
    
    // Reset identification flags but keep model loaded status
    this.identificationService.lastHealthCheck = null;
    this.identificationService.healthCheckInProgress = false;
    
    console.log('✅ Identification-specific state reset');
  }

  // ===================
  // EMERGENCY SHUTDOWN METHODS
  // ===================

  /**
   * Emergency shutdown - force stop everything immediately
   */
  async emergencyShutdown() {
    try {
      console.log('🚨 EMERGENCY SHUTDOWN INITIATED...');
      this.identificationService.setState(IdentificationStates.SHUTTING_DOWN, 'Emergency shutdown');

      // Force stop all streams without waiting
      console.log('⚡ Force stopping all identification streams...');
      const streamIds = Array.from(this.identificationService.currentStreams.keys());
      
      // Don't wait for graceful stops - just clear state
      for (const cameraId of streamIds) {
        try {
          // Try to unfreeze if frozen, but don't wait
          if (this.identificationService.isStreamFrozen(cameraId)) {
            api.post(`/api/detection/detection/identification/stream/${cameraId}/unfreeze`).catch(() => {});
          }
          
          // Try to stop video stream, but don't wait
          api.post(`/api/video_streaming/video/basic/stream/${cameraId}/stop`).catch(() => {});
          
        } catch (error) {
          console.warn(`⚠️ Error in emergency stop for camera ${cameraId}:`, error.message);
        }
      }

      // Force stop all video streams and cameras without waiting
      try {
        console.log('⚡ Force stopping all video streams and cameras...');
        api.post('/api/video_streaming/video/basic/streams/stop_all').catch(() => {});
        api.post("/api/artifact_keeper/camera/stop").catch(() => {});
      } catch (error) {
        console.warn('⚠️ Error in emergency video/camera shutdown:', error.message);
      }

      // Force reset local state immediately
      this.resetLocalState();
      
      // Try backend emergency reset (don't wait for response)
      try {
        api.post('/api/detection/shutdown/graceful', {}, { timeout: 5000 }).catch(() => {
          console.warn('⚠️ Backend emergency shutdown request failed or timed out');
        });
      } catch (error) {
        console.warn('⚠️ Could not send emergency shutdown to backend:', error.message);
      }

      this.identificationService.setState(IdentificationStates.INITIALIZING, 'Emergency shutdown completed');
      
      console.log('🚨 EMERGENCY SHUTDOWN COMPLETED');
      return {
        success: true,
        message: 'Emergency shutdown completed - all services force stopped',
        type: 'emergency'
      };

    } catch (error) {
      console.error('❌ Error during emergency shutdown:', error);
      
      // Even if emergency shutdown fails, reset state
      this.resetLocalState();
      this.identificationService.setState(IdentificationStates.INITIALIZING, 'Emergency shutdown failed - force reset');
      
      return {
        success: false,
        message: 'Emergency shutdown encountered errors but state was reset',
        error: error.message,
        type: 'emergency'
      };
    }
  }

  /**
   * Graceful shutdown with proper cleanup
   */
  async gracefulShutdown() {
    try {
      if (!this.identificationService.canShutdown()) {
        throw new Error(`Cannot shutdown from state: ${this.identificationService.state}`);
      }

      console.log('🛑 Initiating graceful identification system shutdown...');
      this.identificationService.setState(IdentificationStates.SHUTTING_DOWN, 'Graceful shutdown requested');

      // Check if we can shutdown safely
      const safetyCheck = await this.canShutdownSafely();
      if (!safetyCheck.canShutdown) {
        console.warn('⚠️ Shutdown safety check failed, but proceeding...');
      }

      // Get pre-shutdown status for reporting
      const preStatus = await this.getShutdownStatus().catch(() => null);
      
      // Perform the complete shutdown
      const shutdownResult = await this.performCompleteShutdown();
      
      if (shutdownResult.success) {
        console.log('✅ Graceful shutdown completed successfully');
        return {
          ...shutdownResult,
          preShutdownStatus: preStatus,
          type: 'graceful'
        };
      } else {
        throw new Error('Shutdown completed with errors');
      }

    } catch (error) {
      console.error('❌ Error during graceful shutdown:', error);
      
      // Fall back to emergency shutdown if graceful fails
      console.log('🚨 Graceful shutdown failed, attempting emergency shutdown...');
      try {
        const emergencyResult = await this.emergencyShutdown();
        return {
          ...emergencyResult,
          message: `Graceful shutdown failed, emergency shutdown completed: ${emergencyResult.message}`,
          originalError: error.message,
          type: 'emergency_fallback'
        };
      } catch (emergencyError) {
        throw new Error(`Both graceful and emergency shutdown failed: ${error.message} | ${emergencyError.message}`);
      }
    }
  }

  // ===================
  // STATUS AND MONITORING METHODS
  // ===================

/**
 * Monitor shutdown progress - FIXED VERSION
 */
async monitorShutdownProgress(onProgress = null) {
  const startTime = Date.now();
  const maxMonitorTime = this.SHUTDOWN_TIMEOUT + 10000; // Extra buffer
  
  console.log('👀 Starting shutdown progress monitoring...');
  
  return new Promise((resolve) => {
    let checkCount = 0;
    const maxChecks = Math.floor(maxMonitorTime / 2000); // Max number of checks
    
    const checkProgress = async () => {
      try {
        const elapsed = Date.now() - startTime;
        checkCount++;
        
        if (elapsed > maxMonitorTime || checkCount > maxChecks) {
          console.log(`⏰ Shutdown monitoring timeout reached (${elapsed}ms elapsed, ${checkCount} checks)`);
          resolve({
            completed: false,
            timedOut: true,
            elapsed,
            checkCount,
            message: 'Shutdown monitoring timed out'
          });
          return;
        }

        // Get shutdown status from backend
        const status = await this.getShutdownStatus().catch(error => {
          console.warn('⚠️ Status check failed:', error.message);
          return {
            status: 'status_check_failed',
            message: error.message,
            can_shutdown: false
          };
        });
        
        const progress = {
          status: status.status,
          elapsed,
          estimated: (status.estimated_shutdown_time_seconds || 30) * 1000,
          services: status.services,
          frontend: status.frontend,
          step: Math.min(Math.floor(elapsed / 10000) + 1, 4), // Estimate step based on time
          total: 4,
          message: this.getProgressMessage(status.status, elapsed)
        };

        if (onProgress) {
          try {
            onProgress(progress);
          } catch (error) {
            console.warn('⚠️ Error calling onProgress callback:', error);
          }
        }

        console.log(`📊 Shutdown progress check ${checkCount}: ${status.status} (${Math.round(elapsed/1000)}s)`);

        // Check if shutdown is complete
        const backendStopped = status.status === 'all_stopped' || 
                             status.status === 'backend_offline' ||
                             status.status === 'identification_shutdown_complete';
        
        const frontendReady = !status.frontend || 
                             status.frontend.identificationState === 'INITIALIZING' ||
                             status.frontend.identificationState === 'READY';

        if (backendStopped && frontendReady) {
          console.log('✅ Shutdown monitoring completed - all services stopped');
          resolve({
            completed: true,
            elapsed,
            checkCount,
            finalStatus: status,
            message: 'Shutdown completed successfully'
          });
          return;
        }

        // Check for error states
        if (status.status === 'shutdown_error' || status.status === 'error') {
          console.log('❌ Shutdown monitoring detected error state');
          resolve({
            completed: false,
            error: true,
            elapsed,
            checkCount,
            finalStatus: status,
            message: `Shutdown error detected: ${status.message || 'Unknown error'}`
          });
          return;
        }

        // Continue monitoring if not complete
        setTimeout(checkProgress, 2000);

      } catch (error) {
        console.warn(`⚠️ Error during shutdown monitoring check ${checkCount}:`, error.message);
        
        // If we can't get status, assume we need to keep monitoring
        // unless we've exceeded our time/check limits
        if (checkCount < maxChecks && (Date.now() - startTime) < maxMonitorTime) {
          setTimeout(checkProgress, 2000);
        } else {
          resolve({
            completed: false,
            error: true,
            elapsed: Date.now() - startTime,
            checkCount,
            message: `Monitoring failed after ${checkCount} attempts: ${error.message}`
          });
        }
      }
    };

    // Start monitoring
    checkProgress();
  });
}

/**
 * Get appropriate progress message based on status and elapsed time
 */
getProgressMessage(status, elapsed) {
  const seconds = Math.floor(elapsed / 1000);
  
  switch (status) {
    case 'services_running':
      return seconds < 5 ? 'Initiating shutdown...' : 'Stopping services...';
    case 'identification_stopping':
      return 'Stopping identification service...';
    case 'detection_stopping':
      return 'Stopping detection service...';
    case 'cameras_stopping':
      return 'Stopping cameras...';
    case 'all_stopped':
    case 'backend_offline':
      return 'Shutdown complete';
    case 'identification_shutdown_complete':
      return 'Identification shutdown complete';
    default:
      return seconds < 10 ? 'Processing shutdown...' : 'Finalizing shutdown...';
  }
}

  // ===================
  // UTILITY METHODS
  // ===================

  /**
   * Get estimated shutdown time based on current state
   */
  getEstimatedShutdownTime() {
    const baseTime = 5; // Base shutdown time in seconds
    const streamTime = this.identificationService.currentStreams.size * 2; // 2 seconds per stream
    const frozenStreamTime = this.identificationService.frozenStreams.size * 1; // 1 second per frozen stream
    const infrastructureTime = 5; // Additional time for video streams and camera shutdown
    
    return Math.min(baseTime + streamTime + frozenStreamTime + infrastructureTime, 40);
  }

  /**
   * Check if shutdown is currently in progress
   */
  isShutdownInProgress() {
    return this.identificationService.state === IdentificationStates.SHUTTING_DOWN;
  }

  /**
   * Validate shutdown prerequisites
   */
  validateShutdownPrerequisites() {
    const issues = [];
    
    if (!this.identificationService.canShutdown()) {
      issues.push(`Cannot shutdown from current state: ${this.identificationService.state}`);
    }
    
    if (this.identificationService.healthCheckInProgress) {
      issues.push('Health check currently in progress');
    }
    
    if (this.identificationService.initializationPromise) {
      issues.push('Initialization currently in progress');
    }
    
    return {
      canProceed: issues.length === 0,
      issues,
      warnings: this.getShutdownWarnings(
        this.identificationService.currentStreams.size > 0,
        this.identificationService.frozenStreams.size > 0
      )
    };
  }

  // ===================
  // SHUTDOWN OPTIONS AND UTILITIES
  // ===================

  /**
   * Get available shutdown options based on current state
   */
  getShutdownOptions() {
    const options = [];
    
    if (this.identificationService.canShutdown()) {
      options.push({
        id: 'graceful_complete',
        name: 'Complete Graceful Shutdown',
        description: 'Shutdown detection, identification, video streams, and cameras gracefully',
        estimatedTime: this.getEstimatedShutdownTime(),
        recommended: true,
        includes: ['identification service', 'detection service', 'video streams', 'cameras']
      });
      
      options.push({
        id: 'identification_only',
        name: 'Identification Only Shutdown',
        description: 'Shutdown only identification service and its video streams, keep detection running',
        estimatedTime: Math.max(this.getEstimatedShutdownTime() - 10, 5),
        recommended: false,
        includes: ['identification service', 'identification video streams']
      });
    }
    
    // Emergency shutdown is always available
    options.push({
      id: 'emergency',
      name: 'Emergency Shutdown',
      description: 'Force stop all services, video streams, and cameras immediately (may lose data)',
      estimatedTime: 5,
      recommended: false,
      warning: 'This may cause data loss or inconsistent state',
      includes: ['all services', 'all video streams', 'all cameras']
    });

    return options;
  }

  /**
   * Execute shutdown based on option ID
   */
async executeShutdown(optionId, withMonitoring = true) {
  const validationResult = this.validateShutdownPrerequisites();
  
  if (!validationResult.canProceed && optionId !== 'emergency') {
    throw new Error(`Cannot proceed with shutdown: ${validationResult.issues.join(', ')}`);
  }

  console.log(`🚀 Executing shutdown option: ${optionId} (monitoring: ${withMonitoring})`);

  let shutdownPromise;
  
  switch (optionId) {
    case 'graceful_complete':
      console.log('🛑 Starting graceful complete shutdown...');
      shutdownPromise = this.performCompleteShutdown();
      break;
    
    case 'identification_only':
      console.log('🛑 Starting identification-only shutdown...');
      shutdownPromise = this.performIdentificationOnlyShutdown();
      break;
    
    case 'emergency':
      console.log('🚨 Starting emergency shutdown...');
      shutdownPromise = this.emergencyShutdown();
      break;
    
    default:
      throw new Error(`Unknown shutdown option: ${optionId}`);
  }

  if (withMonitoring && optionId !== 'emergency') {
    console.log('👀 Starting shutdown with progress monitoring...');
    
    // Create monitoring promise
    const monitoringPromise = this.monitorShutdownProgress((progress) => {
      console.log(`📊 Shutdown progress: ${progress.status} (${Math.round(progress.elapsed/1000)}s elapsed)`);
    });

    try {
      // Wait for shutdown to complete
      const shutdownResult = await shutdownPromise;
      
      // Give monitoring a moment to detect completion
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      console.log('✅ Shutdown completed successfully:', shutdownResult.message);
      return shutdownResult;
      
    } catch (shutdownError) {
      console.error('❌ Shutdown failed:', shutdownError.message);
      throw shutdownError;
    }
  } else {
    // Execute without monitoring
    console.log('⚡ Executing shutdown without monitoring...');
    return await shutdownPromise;
  }
}
}