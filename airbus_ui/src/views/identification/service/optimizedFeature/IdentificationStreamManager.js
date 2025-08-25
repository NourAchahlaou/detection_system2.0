
import api from "../../../../utils/UseAxios";

// Define the 4 states clearly
const IdentificationStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class IdentificationStreamManager {
  constructor(identificationService) {
    this.identificationService = identificationService;
  }

  // ===================
  // FREEZE/UNFREEZE METHODS
  // ===================

  async freezeStream(cameraId) {
    try {
      console.log(`🧊 Freezing stream for identification on camera ${cameraId}...`);
      
      const response = await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/freeze`);
      
      if (response.data.status === 'frozen') {
        const freezeInfo = {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen',
          purpose: 'identification'
        };
        
        this.identificationService.frozenStreams.set(cameraId, freezeInfo);
        this.identificationService.notifyFreezeListeners(cameraId, 'frozen');
        
        console.log(`✅ Stream frozen for identification on camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to freeze stream for identification');
      }
    } catch (error) {
      console.error(`❌ Error freezing stream for identification on camera ${cameraId}:`, error);
      throw new Error(`Failed to freeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async unfreezeStream(cameraId) {
    try {
      console.log(`🔥 Unfreezing stream after identification on camera ${cameraId}...`);
      
      const response = await api.post(`/api/detection/identification/stream/${cameraId}/unfreeze`);
      
      if (response.data.success) {
        this.identificationService.frozenStreams.delete(cameraId);
        this.identificationService.notifyFreezeListeners(cameraId, 'unfrozen');
        
        console.log(`✅ Stream unfrozen after identification on camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to unfreeze stream after identification');
      }
    } catch (error) {
      console.error(`❌ Error unfreezing stream after identification:`, error);
      throw new Error(`Failed to unfreeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getStreamFreezeStatus(cameraId) {
    try {
      const response = await api.get(`/api/video_streaming/video/basic/stream/${cameraId}/status`);
      
      const isFrozen = response.data.is_frozen || false;
      const streamActive = response.data.stream_active || false;
      
      if (isFrozen && !this.identificationService.frozenStreams.has(cameraId)) {
        // Update local state
        this.identificationService.frozenStreams.set(cameraId, {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen',
          purpose: 'identification'
        });
      } else if (!isFrozen && this.identificationService.frozenStreams.has(cameraId)) {
        this.identificationService.frozenStreams.delete(cameraId);
      }
      
      return {
        cameraId: parseInt(cameraId),
        isFrozen,
        streamActive,
        ...response.data
      };
    } catch (error) {
      console.error(`❌ Error getting freeze status for camera ${cameraId}:`, error);
      return {
        cameraId: parseInt(cameraId),
        isFrozen: false,
        streamActive: false,
        error: error.message
      };
    }
  }

  // ===================
  // IDENTIFICATION METHODS
  // ===================

  async performPieceIdentification(cameraId, options = {}) {
    try {
      const { freezeStream = true, quality = 85 } = options;
      
      console.log(`🔍 Performing piece identification for camera ${cameraId}`);
      
      if (!this.identificationService.isOperational()) {
        throw new Error(`Cannot perform identification in state: ${this.identificationService.state}. Service must be ready.`);
      }

      const requestBody = {
        freeze_stream: freezeStream,
        quality: quality
      };

      const response = await api.post(`/api/detection/identification/identify/${cameraId}`, requestBody);

      if (response.data.success) {
        const identificationData = response.data;
        
        // Update freeze status
        if (identificationData.stream_frozen) {
          this.identificationService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            identificationPerformed: true,
            purpose: 'identification'
          });
          this.identificationService.notifyFreezeListeners(cameraId, 'frozen');
        }

        // Update identification stats
        const streamKey = `identification_${cameraId}`;
        const updatedStats = {
          piecesIdentified: identificationData.summary.total_pieces,
          uniqueLabels: identificationData.summary.unique_labels,
          labelCounts: identificationData.summary.label_counts,
          lastIdentificationTime: Date.now(),
          avgProcessingTime: identificationData.processing_time_ms,
          mode: 'piece_identification',
          streamFrozen: identificationData.stream_frozen
        };

        this.identificationService.identificationStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ Piece identification completed for camera ${cameraId} - Found ${identificationData.summary.total_pieces} pieces`);
        
        return {
          success: true,
          summary: identificationData.summary,
          pieces: identificationData.identification_result.pieces,
          processingTime: identificationData.processing_time_ms,
          frameWithOverlay: identificationData.frame_with_overlay,
          streamFrozen: identificationData.stream_frozen,
          timestamp: identificationData.timestamp,
          message: identificationData.message
        };
      } else {
        throw new Error('Identification request failed');
      }

    } catch (error) {
      console.error("❌ Error performing piece identification:", error);
      throw new Error(`Failed to perform identification: ${error.response?.data?.detail || error.message}`);
    }
  }

  async performQuickAnalysis(cameraId, options = {}) {
    try {
      const { analyzeFrameOnly = false, quality = 85 } = options;
      
      console.log(`🔍 Performing quick analysis for camera ${cameraId}`);
      
      if (!this.identificationService.isOperational()) {
        throw new Error(`Cannot perform analysis in state: ${this.identificationService.state}. Service must be ready.`);
      }

      const requestBody = {
        analyze_frame_only: analyzeFrameOnly,
        quality: quality
      };

      const response = await api.post(`/api/detection/identification/analyze/${cameraId}`, requestBody);

      if (response.data.success) {
        const analysisData = response.data;
        
        // Update identification stats
        const streamKey = `identification_${cameraId}_quick`;
        const updatedStats = {
          piecesFound: analysisData.pieces_found,
          summary: analysisData.summary,
          lastAnalysisTime: Date.now(),
          avgProcessingTime: analysisData.processing_time_ms,
          mode: 'quick_analysis'
        };

        this.identificationService.identificationStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ Quick analysis completed for camera ${cameraId} - ${analysisData.pieces_found} pieces found`);
        
        return {
          success: true,
          piecesFound: analysisData.pieces_found,
          pieces: analysisData.pieces,
          summary: analysisData.summary,
          processingTime: analysisData.processing_time_ms,
          timestamp: analysisData.timestamp,
          message: analysisData.message
        };
      } else {
        throw new Error('Quick analysis request failed');
      }

    } catch (error) {
      console.error("❌ Error performing quick analysis:", error);
      throw new Error(`Failed to perform quick analysis: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getAvailablePieceTypes() {
    try {
      console.log('📋 Getting available piece types...');
      
      const response = await api.get('/api/detection/identification/piece-types');

      if (response.data.success) {
        console.log(`✅ Retrieved ${response.data.total_classes} available piece types`);
        
        return {
          success: true,
          availablePieceTypes: response.data.available_piece_types,
          totalClasses: response.data.total_classes,
          recentlyIdentified: response.data.recently_identified_pieces,
          identificationStats: response.data.identification_stats,
          confidenceThreshold: response.data.confidence_threshold,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get available piece types');
      }

    } catch (error) {
      console.error("❌ Error getting available piece types:", error);
      throw new Error(`Failed to get piece types: ${error.response?.data?.detail || error.message}`);
    }
  }

  async updateConfidenceThreshold(threshold) {
    try {
      console.log(`🎯 Updating confidence threshold to ${threshold}...`);
      
      const response = await api.put('/api/detection/identification/settings/confidence-threshold', {
        threshold: threshold
      });

      if (response.data.success) {
        console.log(`✅ Confidence threshold updated to ${threshold}`);
        
        return {
          success: true,
          newThreshold: response.data.new_threshold,
          message: response.data.message,
          effect: response.data.effect
        };
      } else {
        throw new Error('Failed to update confidence threshold');
      }

    } catch (error) {
      console.error("❌ Error updating confidence threshold:", error);
      throw new Error(`Failed to update threshold: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getIdentificationSettings() {
    try {
      console.log('⚙️ Getting identification settings...');
      
      const response = await api.get('/api/detection/identification/settings');

      if (response.data.success) {
        return {
          success: true,
          settings: response.data.settings,
          capabilities: response.data.capabilities,
          performance: response.data.performance
        };
      } else {
        throw new Error('Failed to get identification settings');
      }

    } catch (error) {
      console.error("❌ Error getting identification settings:", error);
      throw new Error(`Failed to get settings: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getIdentificationHistory() {
    try {
      console.log('📚 Getting identification history...');
      
      const response = await api.get('/api/detection/identification/history');

      if (response.data.success) {
        return {
          success: true,
          recentIdentifications: response.data.recent_identifications,
          totalIdentifications: response.data.total_identifications,
          uniquePiecesIdentified: response.data.unique_pieces_identified,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get identification history');
      }

    } catch (error) {
      console.error("❌ Error getting identification history:", error);
      throw new Error(`Failed to get history: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getIdentificationStats() {
    try {
      console.log('📊 Getting identification statistics...');
      
      const response = await api.get('/api/detection/identification/stats');

      if (response.data.success) {
        return {
          success: true,
          stats: response.data.stats,
          timestamp: response.data.timestamp,
          serviceType: response.data.service_type,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get identification statistics');
      }

    } catch (error) {
      console.error("❌ Error getting identification statistics:", error);
      throw new Error(`Failed to get stats: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // STREAM MANAGEMENT METHODS
  // ===================

  async startIdentificationStream(cameraId, options = {}) {
    try {
      const { streamQuality = 85 } = options;
      console.log(`📺 Starting identification stream for camera ${cameraId}`);

      if (this.identificationService.state !== IdentificationStates.READY) {
        throw new Error(`Cannot start stream in state: ${this.identificationService.state}. Current state must be READY.`);
      }

      await this.ensureCameraStarted(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/basic/stream/${cameraId}?${params}`;
      
      this.identificationService.currentStreams.set(cameraId, {
        url: streamUrl,
        streamKey: `identification_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'identification_stream',
        purpose: 'identification'
      });

      if (this.identificationService.state === IdentificationStates.READY) {
        this.identificationService.setState(IdentificationStates.RUNNING, `Started identification stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started identification stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting identification stream:", error);
      throw new Error(`Failed to start identification stream: ${error.message}`);
    }
  }

  async stopIdentificationStream(cameraId) {
    try {
      const stream = this.identificationService.currentStreams.get(cameraId);
      if (stream && stream.type === 'identification_stream') {
        console.log(`⏹️ Stopping identification stream for camera ${cameraId}`);
        
        // Unfreeze stream if it's frozen
        if (this.identificationService.isStreamFrozen(cameraId)) {
          try {
            await this.unfreezeStream(cameraId);
          } catch (error) {
            console.warn(`⚠️ Error unfreezing stream during stop for camera ${cameraId}:`, error.message);
          }
        }
        
        this.stopStatsMonitoring(cameraId);
        
        try {
          await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/stop`);
          await api.post("/api/artifact_keeper/camera/stop");
        } catch (error) {
          console.warn(`⚠️ Error stopping identification stream API for camera ${cameraId}:`, error.message);
        }
        
        this.identificationService.currentStreams.delete(cameraId);
        
        // Update state based on remaining streams
        if (this.identificationService.currentStreams.size === 0 && this.identificationService.state === IdentificationStates.RUNNING) {
          this.identificationService.setState(IdentificationStates.READY, 'All identification streams stopped');
        }
        
        console.log(`✅ Successfully stopped identification stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping identification stream:", error);
      throw error;
    }
  }

  async stopAllIdentificationStreams(performShutdown = true) {
    try {
      console.log('🛑 Stopping all identification streams...');
      
      const stopPromises = Array.from(this.identificationService.currentStreams.keys()).map(cameraId => {
        return this.stopIdentificationStream(cameraId);
      });
      
      await Promise.allSettled(stopPromises);
      
      // Unfreeze all frozen streams
      const frozenStreams = this.identificationService.getFrozenStreams();
      for (const frozenStream of frozenStreams) {
        try {
          await this.unfreezeStream(frozenStream.cameraId);
        } catch (error) {
          console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId}:`, error.message);
        }
      }
      
      this.identificationService.currentStreams.clear();
      this.identificationService.identificationStats.clear();
      for (const cameraId of this.identificationService.eventListeners.keys()) {
        this.stopStatsMonitoring(cameraId);
      }
      
      if (this.identificationService.state === IdentificationStates.RUNNING) {
        this.identificationService.setState(IdentificationStates.READY, 'All identification streams stopped');
      }
      
      if (performShutdown) {
        try {
          console.log('🔄 Performing graceful identification shutdown after stopping all streams...');
          await this.identificationService.gracefulShutdown();
          console.log('✅ All identification streams stopped and service gracefully shut down');
        } catch (shutdownError) {
          console.error('⚠️ Graceful shutdown failed, but all streams stopped:', shutdownError.message);
        }
      }
      
      console.log("✅ Stopped all identification streams");
    } catch (error) {
      console.error("❌ Error stopping all identification streams:", error);
      throw error;
    }
  }

  async ensureCameraStarted(cameraId) {
    try {
      const numericCameraId = parseInt(cameraId, 10);
      if (isNaN(numericCameraId)) {
        throw new Error(`Invalid camera ID: ${cameraId}`);
      }

      console.log(`📹 Starting camera ${numericCameraId} for identification...`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.identificationService.CAMERA_START_TIMEOUT);

      try {
        const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
          camera_id: numericCameraId
        }, {
          signal: controller.signal,
          timeout: this.identificationService.CAMERA_START_TIMEOUT
        });
        
        clearTimeout(timeoutId);
        console.log("✅ Camera started for identification:", cameraResponse.data.message);

        await new Promise(resolve => setTimeout(resolve, 2000));

        return {
          success: true,
          message: cameraResponse.data.message
        };

      } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
          throw new Error(`Camera startup timed out after ${this.identificationService.CAMERA_START_TIMEOUT / 1000} seconds`);
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

  // ===================
  // STATS MONITORING METHODS
  // ===================

  startStatsMonitoring = (cameraId) => {
    const monitorInterval = setInterval(async () => {
      if (this.identificationService.state === IdentificationStates.SHUTTING_DOWN) {
        clearInterval(monitorInterval);
        return;
      }

      try {
        // Check stream status and freeze state
        const freezeStatus = await this.getStreamFreezeStatus(cameraId);
        
        // Update stats based on stream status
        const streamKey = `identification_${cameraId}`;
        const currentStats = this.identificationService.identificationStats.get(streamKey) || {};
        
        const updatedStats = {
          ...currentStats,
          streamActive: freezeStatus.streamActive,
          isFrozen: freezeStatus.isFrozen,
          lastStatusCheck: Date.now(),
          mode: 'identification_monitoring'
        };

        this.identificationService.identificationStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);
        
      } catch (error) {
        if (this.identificationService.state !== IdentificationStates.SHUTTING_DOWN) {
          console.debug("Error in identification monitoring:", error);
        }
      }
    }, 5000); // Check every 5 seconds

    // Store interval for cleanup
    if (!this.identificationService.eventListeners.has(cameraId)) {
      this.identificationService.eventListeners.set(cameraId, { monitorInterval, listeners: [] });
    } else {
      this.identificationService.eventListeners.get(cameraId).monitorInterval = monitorInterval;
    }
  };

  stopStatsMonitoring = (cameraId) => {
    const eventData = this.identificationService.eventListeners.get(cameraId);
    if (eventData?.monitorInterval) {
      clearInterval(eventData.monitorInterval);
    }
    this.identificationService.eventListeners.delete(cameraId);
  };

  addStatsListener = (cameraId, callback) => {
    if (!this.identificationService.eventListeners.has(cameraId)) {
      this.identificationService.eventListeners.set(cameraId, { listeners: [] });
    }
    this.identificationService.eventListeners.get(cameraId).listeners.push(callback);
  };

  removeStatsListener = (cameraId, callback) => {
    const eventData = this.identificationService.eventListeners.get(cameraId);
    if (eventData) {
      eventData.listeners = eventData.listeners.filter(cb => cb !== callback);
    }
  };

  notifyStatsListeners = (cameraId, stats) => {
    const eventData = this.identificationService.eventListeners.get(cameraId);
    if (eventData?.listeners) {
      eventData.listeners.forEach(callback => {
        try {
          callback(stats);
        } catch (error) {
          console.error("Error in identification stats listener callback:", error);
        }
      });
    }
  };

  async getAllIdentificationStreamingStats() {
    try {
      if (this.identificationService.state === IdentificationStates.SHUTTING_DOWN) {
        return {
          active_streams: 0,
          avg_processing_time_ms: 0,
          total_identifications: 0,
          system_load_percent: 0,
          memory_usage_mb: 0,
          frozen_streams: this.identificationService.getFrozenStreams().length,
          mode: 'identification'
        };
      }

      const response = await api.get('/api/video_streaming/video/basic/stats');
      
      // Add freeze information for identification mode
      response.data.frozen_streams = this.identificationService.getFrozenStreams().length;
      response.data.live_streams = (response.data.total_active_streams || 0) - response.data.frozen_streams;
      
      return { ...response.data, mode: 'identification' };
    } catch (error) { 
      console.error("❌ Error getting identification streaming stats:", error);
      throw error;
    }
  }

  cleanup = async () => { 
    try {
      console.log('🧹 Starting IdentificationStreamManager cleanup...');
      
      // Stop all identification streams
      const identificationStreams = Array.from(this.identificationService.currentStreams.entries())
        .filter(([_, stream]) => stream.type === 'identification_stream');
      
      const stopPromises = identificationStreams.map(([cameraId, _]) => this.stopIdentificationStream(cameraId));
      await Promise.allSettled(stopPromises);
      
      // Unfreeze all frozen streams
      const frozenStreams = this.identificationService.getFrozenStreams();
      for (const frozenStream of frozenStreams) {
        try {
          await this.unfreezeStream(frozenStream.cameraId);
        } catch (error) {
          console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId}:`, error.message);
        }
      }
      
      console.log('✅ IdentificationStreamManager cleanup completed');
    } catch (error) {
      console.error("❌ Error during IdentificationStreamManager cleanup:", error);
    }
  };
}