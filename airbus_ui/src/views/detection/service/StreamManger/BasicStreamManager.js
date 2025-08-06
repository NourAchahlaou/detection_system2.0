import api from "../../../../utils/UseAxios";

// Define the 4 states clearly
const DetectionStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY', 
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class BasicStreamManager {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  // ===================
  // FREEZE/UNFREEZE METHODS
  // ===================

  async freezeStream(cameraId) {
    try {
      console.log(`🧊 Freezing stream for camera ${cameraId}...`);
      
      const response = await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/freeze`);
      
      if (response.data.status === 'frozen') {
        const freezeInfo = {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen'
        };
        
        this.detectionService.frozenStreams.set(cameraId, freezeInfo);
        this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        
        console.log(`✅ Stream frozen for camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to freeze stream');
      }
    } catch (error) {
      console.error(`❌ Error freezing stream for camera ${cameraId}:`, error);
      throw new Error(`Failed to freeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async unfreezeStream(cameraId) {
    try {
      console.log(`🔥 Unfreezing stream for camera ${cameraId}...`);
      
      const response = await api.post(`/api/video_streaming/video/basic/stream/${cameraId}/unfreeze`);
      
      if (response.data.status === 'unfrozen') {
        this.detectionService.frozenStreams.delete(cameraId);
        this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
        
        console.log(`✅ Stream unfrozen for camera ${cameraId}`);
        return { success: true, ...response.data };
      } else {
        throw new Error('Failed to unfreeze stream');
      }
    } catch (error) {
      console.error(`❌ Error unfreezing stream for camera ${cameraId}:`, error);
      throw new Error(`Failed to unfreeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getStreamFreezeStatus(cameraId) {
    try {
      const response = await api.get(`/api/video_streaming/video/basic/stream/${cameraId}/status`);
      
      const isFrozen = response.data.is_frozen || false;
      const streamActive = response.data.stream_active || false;
      
      if (isFrozen && !this.detectionService.frozenStreams.has(cameraId)) {
        // Update local state
        this.detectionService.frozenStreams.set(cameraId, {
          cameraId: parseInt(cameraId),
          frozenAt: Date.now(),
          status: 'frozen'
        });
      } else if (!isFrozen && this.detectionService.frozenStreams.has(cameraId)) {
        this.detectionService.frozenStreams.delete(cameraId);
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
  // ENHANCED DETECTION METHODS WITH DATABASE INTEGRATION
  // ===================

  async createDetectionLot(lotName, expectedPieceId, expectedPieceNumber) {
    try {
      console.log(`📦 Creating detection lot: '${lotName}' expecting piece ${expectedPieceNumber}`);
      
      const response = await api.post('/api/detection/basic/lots', {
        lot_name: lotName,
        expected_piece_id: expectedPieceId,
        expected_piece_number: expectedPieceNumber
      });

      if (response.data.success) {
        console.log(`✅ Created lot ${response.data.lot_id}: '${lotName}'`);
        return {
          success: true,
          lotId: response.data.lot_id,
          lotData: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to create detection lot');
      }
    } catch (error) {
      console.error(`❌ Error creating detection lot:`, error);
      throw new Error(`Failed to create lot: ${error.response?.data?.detail || error.message}`);
    }
  }

  // FIXED: getDetectionLot method with consistent property naming
  async getDetectionLot(lotId) {
    try {
      console.log(`📦 Getting detection lot ${lotId}`);
      
      const response = await api.get(`/api/detection/basic/lots/${lotId}`);

      if (response.data.success) {
        // FIXED: Return both 'lot' and 'lotData' for backward compatibility
        return {
          success: true,
          lot: response.data.data,        // For consistency with frontend expectations
          lotData: response.data.data,    // For backward compatibility
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get detection lot');
      }
    } catch (error) {
      console.error(`❌ Error getting detection lot ${lotId}:`, error);
      
      // FIXED: Better error handling
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message;
      
      return {
        success: false,
        lot: null,
        lotData: null,
        message: errorMessage,
        error: errorMessage
      };
    }
  }

  async updateLotTargetMatchStatus(lotId, isTargetMatch) {
    try {
      console.log(`📦 Updating lot ${lotId} target match status: ${isTargetMatch}`);
      
      const response = await api.put(`/api/detection/basic/lots/${lotId}/status`, {
        is_target_match: isTargetMatch
      });

      if (response.data.success) {
        console.log(`✅ Updated lot ${lotId} status: ${isTargetMatch ? 'completed' : 'needs correction'}`);
        return {
          success: true,
          lotData: response.data.data,
          isTargetMatch: response.data.is_target_match,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to update lot status');
      }
    } catch (error) {
      console.error(`❌ Error updating lot ${lotId} status:`, error);
      throw new Error(`Failed to update lot status: ${error.response?.data?.detail || error.message}`);
    }
  }
  async getAllDetectionLots() {
    try {
      console.log(`📦 Getting all detection lots`)
      const response = await api.get('/api/detection/basic/all_lots');

      if (response.data.success) {
        return {
          success: true,
          lots: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get all detection lots');
      }
    } catch (error) {
      console.error(`❌ Error getting all detection lots:`, error);
      throw new Error(`Failed to get all lots: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getLotDetectionSessions(lotId) {
    try {
      console.log(`📊 Getting detection sessions for lot ${lotId}`);
      
      const response = await api.get(`/api/detection/basic/lots/${lotId}/sessions`);

      if (response.data.success) {
        return {
          success: true,
          lotId: response.data.lot_id,
          lotName: response.data.lot_name,
          totalSessions: response.data.total_sessions,
          successfulDetections: response.data.successful_detections,
          sessions: response.data.sessions,
          lotStatus: response.data.lot_status
        };
      } else {
        throw new Error('Failed to get lot sessions');
      }
    } catch (error) {
      console.error(`❌ Error getting lot ${lotId} sessions:`, error);
      throw new Error(`Failed to get lot sessions: ${error.response?.data?.detail || error.message}`);
    }
  }

  async performDetectionWithLotTracking(cameraId, targetLabel, options = {}) {
    try {
      const { 
        lotId = null, 
        expectedPieceId = null, 
        expectedPieceNumber = null, 
        quality = 85 
      } = options;
      
      console.log(`🎯 Performing lot-tracked detection for camera ${cameraId}, target: '${targetLabel}'${lotId ? `, lot: ${lotId}` : ''}`);
      
      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      const requestBody = {
        target_label: targetLabel,
        lot_id: lotId,
        expected_piece_id: expectedPieceId,
        expected_piece_number: expectedPieceNumber,
        quality: quality
      };

      const response = await api.post(`/api/detection/basic/detect/${cameraId}`, requestBody);

      if (response.data.success) {
        const detectionData = response.data.data;
        
        // Update freeze status
        if (detectionData.stream_frozen) {
          this.detectionService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            detectionPerformed: true,
            lotId: detectionData.lot_id,
            sessionId: detectionData.session_id
          });
          this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        }

        // Update detection stats with enhanced data
        const streamKey = `enhanced_${cameraId}_${targetLabel}`;
        const updatedStats = {
          objectDetected: detectionData.detected_target,
          detectionCount: detectionData.detected_target ? 1 : 0,
          lastDetectionTime: detectionData.detected_target ? Date.now() : null,
          avgProcessingTime: detectionData.processing_time_ms,
          confidence: detectionData.confidence,
          mode: 'enhanced_lot_tracking',
          streamFrozen: detectionData.stream_frozen,
          lotId: detectionData.lot_id,
          sessionId: detectionData.session_id,
          isTargetMatch: detectionData.is_target_match,
          detectionRate: detectionData.detection_rate
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ Lot-tracked detection completed for camera ${cameraId} - Target detected: ${detectionData.detected_target}`);
        
        return {
          success: true,
          detected: detectionData.detected_target,
          confidence: detectionData.confidence,
          processingTime: detectionData.processing_time_ms,
          frameWithOverlay: detectionData.frame_with_overlay,
          streamFrozen: detectionData.stream_frozen,
          timestamp: detectionData.timestamp,
          lotId: detectionData.lot_id,
          sessionId: detectionData.session_id,
          isTargetMatch: detectionData.is_target_match,
          detectionRate: detectionData.detection_rate,
          message: response.data.message
        };
      } else {
        throw new Error('Enhanced detection request failed');
      }

    } catch (error) {
      console.error("❌ Error performing lot-tracked detection:", error);
      throw new Error(`Failed to perform lot-tracked detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  async createLotAndDetect(cameraId, lotName, expectedPieceId, expectedPieceNumber, targetLabel, options = {}) {
    try {
      const { quality = 85 } = options;
      
      console.log(`🚀 Creating lot and detecting for camera ${cameraId}`);
      
      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      const requestBody = {
        lot_name: lotName,
        expected_piece_id: expectedPieceId,
        expected_piece_number: expectedPieceNumber,
        target_label: targetLabel,
        quality: quality
      };

      const response = await api.post(`/api/detection/basic/detect/${cameraId}/with-lot-creation`, requestBody);

      if (response.data.success) {
        const lotData = response.data.lot_created;
        const detectionData = response.data.detection_result;
        
        // Update freeze status
        if (detectionData.stream_frozen) {
          this.detectionService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            detectionPerformed: true,
            lotId: lotData.lot_id,
            sessionId: detectionData.session_id
          });
          this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        }

        // Update detection stats
        const streamKey = `enhanced_${cameraId}_${targetLabel}`;
        const updatedStats = {
          objectDetected: detectionData.detected_target,
          detectionCount: detectionData.detected_target ? 1 : 0,
          lastDetectionTime: detectionData.detected_target ? Date.now() : null,
          avgProcessingTime: detectionData.processing_time_ms,
          confidence: detectionData.confidence,
          mode: 'enhanced_lot_creation',
          streamFrozen: detectionData.stream_frozen,
          lotId: lotData.lot_id,
          sessionId: detectionData.session_id,
          isTargetMatch: detectionData.is_target_match,
          detectionRate: detectionData.detection_rate
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ Lot created and detection completed for camera ${cameraId}`);
        
        return {
          success: true,
          lotCreated: lotData,
          detectionResult: {
            detected: detectionData.detected_target,
            confidence: detectionData.confidence,
            processingTime: detectionData.processing_time_ms,
            frameWithOverlay: detectionData.frame_with_overlay,
            streamFrozen: detectionData.stream_frozen,
            timestamp: detectionData.timestamp,
            lotId: detectionData.lot_id,
            sessionId: detectionData.session_id,
            isTargetMatch: detectionData.is_target_match,
            detectionRate: detectionData.detection_rate
          },
          message: response.data.message
        };
      } else {
        throw new Error('Lot creation and detection failed');
      }

    } catch (error) {
      console.error("❌ Error creating lot and detecting:", error);
      throw new Error(`Failed to create lot and detect: ${error.response?.data?.detail || error.message}`);
    }
  }

  async performDetectionWithAutoCorrection(cameraId, lotId, targetLabel, options = {}) {
    try {
      const { quality = 85, autoCompleteOnMatch = true } = options;
      
      console.log(`🔄 Performing auto-correction detection for camera ${cameraId}, lot: ${lotId}`);
      
      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      const requestBody = {
        lot_id: lotId,
        target_label: targetLabel,
        quality: quality,
        auto_complete_on_match: autoCompleteOnMatch
      };

      const response = await api.post(`/api/detection/basic/detect/${cameraId}/with-auto-correction`, requestBody);

      if (response.data.success) {
        const detectionData = response.data.detection_result;
        
        // Update freeze status
        if (detectionData?.stream_frozen) {
          this.detectionService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            detectionPerformed: true,
            lotId: detectionData.lot_id,
            sessionId: detectionData.session_id
          });
          this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        }

        // Update detection stats
        const streamKey = `enhanced_${cameraId}_${targetLabel}`;
        const updatedStats = {
          objectDetected: detectionData?.detected_target || false,
          detectionCount: detectionData?.detected_target ? 1 : 0,
          lastDetectionTime: detectionData?.detected_target ? Date.now() : null,
          avgProcessingTime: detectionData?.processing_time_ms || 0,
          confidence: detectionData?.confidence,
          mode: 'enhanced_auto_correction',
          streamFrozen: detectionData?.stream_frozen || false,
          lotId: detectionData?.lot_id,
          sessionId: detectionData?.session_id,
          isTargetMatch: detectionData?.is_target_match || false,
          detectionRate: detectionData?.detection_rate || 0,
          correctionAction: response.data.correction_action,
          lotCompleted: response.data.lot_completed
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ Auto-correction detection completed: ${response.data.correction_action}`);
        
        return {
          success: true,
          detected: detectionData?.detected_target || false,
          confidence: detectionData?.confidence,
          processingTime: detectionData?.processing_time_ms || 0,
          frameWithOverlay: detectionData?.frame_with_overlay || '',
          streamFrozen: detectionData?.stream_frozen || false,
          timestamp: detectionData?.timestamp || Date.now(),
          lotId: detectionData?.lot_id,
          sessionId: detectionData?.session_id,
          isTargetMatch: detectionData?.is_target_match || false,
          detectionRate: detectionData?.detection_rate || 0,
          correctionAction: response.data.correction_action,
          lotCompleted: response.data.lot_completed,
          message: response.data.message,
          detectionSkipped: response.data.detection_skipped || false
        };
      } else {
        throw new Error('Auto-correction detection failed');
      }

    } catch (error) {
      console.error("❌ Error performing auto-correction detection:", error);
      throw new Error(`Failed to perform auto-correction detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  async unfreezeStreamAfterDetection(cameraId) {
    try {
      console.log(`🔥 Unfreezing stream after detection for camera ${cameraId}`);
      
      const response = await api.post(`/api/detection/basic/stream/${cameraId}/unfreeze`);

      if (response.data.success) {
        this.detectionService.frozenStreams.delete(cameraId);
        this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
        
        console.log(`✅ Stream unfrozen after detection for camera ${cameraId}`);
        return {
          success: true,
          cameraId: response.data.camera_id,
          message: response.data.message,
          timestamp: response.data.timestamp
        };
      } else {
        throw new Error('Failed to unfreeze stream after detection');
      }
    } catch (error) {
      console.error(`❌ Error unfreezing stream after detection:`, error);
      throw new Error(`Failed to unfreeze stream: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // BASIC DETECTION METHODS (Original methods kept intact)
  // ===================

  async performOnDemandDetection(cameraId, targetLabel, options = {}) {
    try {
      const { quality = 85, autoUnfreeze = false, unfreezeDelay = 2.0 } = options;
      
      console.log(`🎯 Performing on-demand detection for camera ${cameraId}, target: '${targetLabel}'`);
      
      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      // Choose detection endpoint based on auto-unfreeze option
      const endpoint = autoUnfreeze 
        ? `/api/detection/basic/detect/${cameraId}/with-unfreeze`
        : `/api/detection/basic/detect/${cameraId}`;

      const requestBody = {
        target_label: targetLabel,
        quality: quality
      };

      if (autoUnfreeze) {
        requestBody.unfreeze_delay = unfreezeDelay;
      }

      const response = await api.post(endpoint, requestBody);

      if (response.data.success) {
        const detectionData = response.data.data;
        
        // Update freeze status
        if (detectionData.stream_frozen && !autoUnfreeze) {
          this.detectionService.frozenStreams.set(cameraId, {
            cameraId: parseInt(cameraId),
            frozenAt: Date.now(),
            status: 'frozen',
            detectionPerformed: true
          });
          this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
        } else if (autoUnfreeze) {
          this.detectionService.frozenStreams.delete(cameraId);
          this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
        }

        // Update detection stats
        const streamKey = `basic_${cameraId}_${targetLabel}`;
        const updatedStats = {
          objectDetected: detectionData.detected_target,
          detectionCount: detectionData.detected_target ? 1 : 0,
          lastDetectionTime: detectionData.detected_target ? Date.now() : null,
          avgProcessingTime: detectionData.processing_time_ms,
          confidence: detectionData.confidence,
          mode: 'basic_on_demand',
          streamFrozen: detectionData.stream_frozen,
          autoUnfrozen: autoUnfreeze
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);

        console.log(`✅ On-demand detection completed for camera ${cameraId}${autoUnfreeze ? ' (auto-unfrozen)' : ' (stream frozen)'}`);
        
        return {
          success: true,
          detected: detectionData.detected_target,
          confidence: detectionData.confidence,
          processingTime: detectionData.processing_time_ms,
          frameWithOverlay: detectionData.frame_with_overlay,
          streamFrozen: detectionData.stream_frozen,
          autoUnfrozen: autoUnfreeze,
          timestamp: detectionData.timestamp
        };
      } else {
        throw new Error('Detection request failed');
      }

    } catch (error) {
      console.error("❌ Error performing on-demand detection:", error);
      throw new Error(`Failed to perform detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  async performBatchDetection(detections = []) {
    try {
      console.log(`🎯 Performing batch detection on ${detections.length} streams`);

      if (!this.detectionService.isOperational()) {
        throw new Error(`Cannot perform batch detection in state: ${this.detectionService.state}. Service must be ready.`);
      }

      if (detections.length === 0) {
        throw new Error('No detections provided');
      }

      if (detections.length > 3) {
        throw new Error('Maximum 3 detections per batch in basic mode');
      }

      const response = await api.post('/api/detection/basic/detect/batch', {
        detections: detections
      });

      if (response.data.success) {
        const results = response.data.results;
        
        // Update local state for each detection
        results.forEach(result => {
          if (result.success && result.data) {
            const cameraId = result.camera_id;
            const detectionData = result.data;
            
            // Update freeze status
            if (detectionData.stream_frozen && !result.auto_unfrozen) {
              this.detectionService.frozenStreams.set(cameraId, {
                cameraId: parseInt(cameraId),
                frozenAt: Date.now(),
                status: 'frozen',
                detectionPerformed: true
              });
              this.detectionService.notifyFreezeListeners(cameraId, 'frozen');
            } else if (result.auto_unfrozen) {
              this.detectionService.frozenStreams.delete(cameraId);
              this.detectionService.notifyFreezeListeners(cameraId, 'unfrozen');
            }

            // Update detection stats
            const streamKey = `basic_${cameraId}_batch`;
            const updatedStats = {
              objectDetected: detectionData.detected_target,
              detectionCount: detectionData.detected_target ? 1 : 0,
              lastDetectionTime: detectionData.detected_target ? Date.now() : null,
              avgProcessingTime: detectionData.processing_time_ms,
              confidence: detectionData.confidence,
              mode: 'basic_batch',
              streamFrozen: detectionData.stream_frozen,
              autoUnfrozen: result.auto_unfrozen
            };

            this.detectionService.detectionStats.set(streamKey, updatedStats);
            this.notifyStatsListeners(cameraId, updatedStats);
          }
        });

        console.log(`✅ Batch detection completed: ${response.data.successful}/${response.data.total_processed} successful`);
        return response.data;
      } else {
        throw new Error('Batch detection request failed');
      }

    } catch (error) {
      console.error("❌ Error performing batch detection:", error);
      throw new Error(`Failed to perform batch detection: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // BASIC STREAMING METHODS (Original methods kept intact)
  // ===================

  async startBasicDetectionFeed(cameraId, targetLabel, options = {}) {
    try {
      console.log(`🎯 Starting basic detection feed for camera ${cameraId} with target: ${targetLabel}`);

      if (!this.detectionService.canStart()) {
        throw new Error(`Cannot start detection in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      // Start camera
      await this.ensureCameraStarted(cameraId);

      // Store stream info
      const streamKey = `basic_${cameraId}_${targetLabel}`;
      this.detectionService.currentStreams.set(cameraId, {
        url: `/api/video_streaming/video/basic/stream/${cameraId}`,
        targetLabel,
        streamKey,
        startTime: Date.now(),
        isActive: true,
        type: 'basic_detection'
      });

      // Update state to running
      this.detectionService.setState(DetectionStates.RUNNING, `Started basic detection for camera ${cameraId}`);

      // Start detection polling for basic mode (but less frequently since it's on-demand)
      this.startBasicDetectionMonitoring(cameraId, targetLabel);

      console.log(`✅ Successfully started basic detection feed for camera ${cameraId}`);
      return `/api/video_streaming/video/basic/stream/${cameraId}`;

    } catch (error) {
      console.error("❌ Error starting basic detection feed:", error);
      throw new Error(`Failed to start basic detection feed: ${error.message}`);
    }
  }

  async startBasicStream(cameraId, options = {}) {
    try {
      const { streamQuality = 85 } = options;
      console.log(`📺 Starting basic stream for camera ${cameraId}`);

      // Fix: Use explicit state check instead of canStart()
      if (this.detectionService.state !== DetectionStates.READY) {
        throw new Error(`Cannot start stream in state: ${this.detectionService.state}. Current state must be READY.`);
      }

      await this.ensureCameraStarted(cameraId);

      const params = new URLSearchParams({
        stream_quality: streamQuality.toString()
      });

      const streamUrl = `/api/video_streaming/video/basic/stream/${cameraId}?${params}`;
      
      this.detectionService.currentStreams.set(cameraId, {
        url: streamUrl,
        targetLabel: null,
        streamKey: `basic_stream_${cameraId}`,
        startTime: Date.now(),
        isActive: true,
        type: 'basic_stream'
      });

      if (this.detectionService.state === DetectionStates.READY) {
        this.detectionService.setState(DetectionStates.RUNNING, `Started basic stream for camera ${cameraId}`);
      }

      console.log(`✅ Successfully started basic stream for camera ${cameraId}`);
      return streamUrl;

    } catch (error) {
      console.error("❌ Error starting basic stream:", error);
      throw new Error(`Failed to start basic stream: ${error.message}`);
    }
  }

  async stopBasicStream(cameraId) {
    try {
      const stream = this.detectionService.currentStreams.get(cameraId);
      if (stream && stream.type && stream.type.startsWith('basic')) {
        console.log(`⏹️ Stopping basic stream for camera ${cameraId}`);
        
        // Unfreeze stream if it's frozen
        if (this.detectionService.isStreamFrozen(cameraId)) {
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
          console.warn(`⚠️ Error stopping basic stream API for camera ${cameraId}:`, error.message);
        }
        
        this.detectionService.currentStreams.delete(cameraId);
        
        // Update state based on remaining streams
        if (this.detectionService.currentStreams.size === 0 && this.detectionService.state === DetectionStates.RUNNING) {
          this.detectionService.setState(DetectionStates.READY, 'All streams stopped');
        }
        
        console.log(`✅ Successfully stopped basic stream for camera ${cameraId}`);
      }
    } catch (error) {
      console.error("❌ Error stopping basic stream:", error);
      throw error;
    }
  }

  async startBasicDetectionMonitoring(cameraId, targetLabel) {
    // For basic mode, we don't need continuous polling since detection is on-demand
    // Just set up monitoring for stream status and freeze state
    const monitorInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        clearInterval(monitorInterval);
        return;
      }

      try {
        // Check stream status and freeze state
        const freezeStatus = await this.getStreamFreezeStatus(cameraId);
        
        // Update stats based on stream status
        const streamKey = `basic_${cameraId}_${targetLabel}`;
        const currentStats = this.detectionService.detectionStats.get(streamKey) || {};
        
        const updatedStats = {
          ...currentStats,
          streamActive: freezeStatus.streamActive,
          isFrozen: freezeStatus.isFrozen,
          lastStatusCheck: Date.now(),
          mode: 'basic_monitoring'
        };

        this.detectionService.detectionStats.set(streamKey, updatedStats);
        this.notifyStatsListeners(cameraId, updatedStats);
        
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error in basic detection monitoring:", error);
        }
      }
    }, 5000); // Check every 5 seconds (less frequent than continuous detection)

    // Store interval for cleanup
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { monitorInterval, listeners: [] });
    } else {
      this.detectionService.eventListeners.get(cameraId).monitorInterval = monitorInterval;
    }
  }

  async startBasicDetectionPolling(cameraId, targetLabel) {
    const pollInterval = setInterval(async () => {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        clearInterval(pollInterval);
        return;
      }

      try {
        // Perform detection on single frame
        const response = await api.post(`/api/detection/basic/detect/${cameraId}`, {
          target_label: targetLabel,
          quality: 85
        });

        if (response.data.success) {
          const detectionData = response.data.data;
          const streamKey = `basic_${cameraId}_${targetLabel}`;
          
          // Update detection stats
          const currentStats = this.detectionService.detectionStats.get(streamKey) || {};
          const updatedStats = {
            ...currentStats,
            objectDetected: detectionData.target_detected,
            detectionCount: detectionData.target_detected ? (currentStats.detectionCount || 0) + 1 : (currentStats.detectionCount || 0),
            lastDetectionTime: detectionData.target_detected ? Date.now() : currentStats.lastDetectionTime,
            avgProcessingTime: detectionData.processing_time_ms,
            confidence: detectionData.confidence
          };

          this.detectionService.detectionStats.set(streamKey, updatedStats);
          this.notifyStatsListeners(cameraId, updatedStats);
        }
      } catch (error) {
        if (this.detectionService.state !== DetectionStates.SHUTTING_DOWN) {
          console.debug("Error in basic detection polling:", error);
        }
      }
    }, 2000); // Poll every 2 seconds for basic mode

    // Store interval for cleanup
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { pollInterval, listeners: [] });
    } else {
      this.detectionService.eventListeners.get(cameraId).pollInterval = pollInterval;
    }
  }

  // ===================
  // SHARED UTILITY METHODS
  // ===================

  async ensureCameraStarted(cameraId) {
    try {
      const numericCameraId = parseInt(cameraId, 10);
      if (isNaN(numericCameraId)) {
        throw new Error(`Invalid camera ID: ${cameraId}`);
      }

      console.log(`📹 Starting camera ${numericCameraId} for detection...`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.detectionService.CAMERA_START_TIMEOUT);

      try {
        const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
          camera_id: numericCameraId
        }, {
          signal: controller.signal,
          timeout: this.detectionService.CAMERA_START_TIMEOUT
        });
        
        clearTimeout(timeoutId);
        console.log("✅ Camera started for detection:", cameraResponse.data.message);

        await new Promise(resolve => setTimeout(resolve, 2000));

        return {
          success: true,
          message: cameraResponse.data.message
        };

      } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
          throw new Error(`Camera startup timed out after ${this.detectionService.CAMERA_START_TIMEOUT / 1000} seconds`);
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

  // Stats monitoring methods
  stopStatsMonitoring = (cameraId) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData?.pollInterval) {
      clearInterval(eventData.pollInterval);
    }
    if (eventData?.monitorInterval) {
      clearInterval(eventData.monitorInterval);
    }
    this.detectionService.eventListeners.delete(cameraId);
  };

  addStatsListener = (cameraId, callback) => {
    if (!this.detectionService.eventListeners.has(cameraId)) {
      this.detectionService.eventListeners.set(cameraId, { listeners: [] });
    }
    this.detectionService.eventListeners.get(cameraId).listeners.push(callback);
  };

  removeStatsListener = (cameraId, callback) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData) {
      eventData.listeners = eventData.listeners.filter(cb => cb !== callback);
    }
  };

  notifyStatsListeners = (cameraId, stats) => {
    const eventData = this.detectionService.eventListeners.get(cameraId);
    if (eventData?.listeners) {
      eventData.listeners.forEach(callback => {
        try {
          callback(stats);
        } catch (error) {
          console.error("Error in stats listener callback:", error);
        }
      });
    }
  };

  async getAllBasicStreamingStats() {
    try {
      if (this.detectionService.state === DetectionStates.SHUTTING_DOWN) {
        return {
          active_streams: 0,
          avg_processing_time_ms: 0,
          total_detections: 0,
          system_load_percent: 0,
          memory_usage_mb: 0,
          frozen_streams: this.detectionService.getFrozenStreams().length,
          mode: 'basic'
        };
      }

      const response = await api.get('/api/video_streaming/video/basic/stats');
      
      // Add freeze information for basic mode
      response.data.frozen_streams = this.detectionService.getFrozenStreams().length;
      response.data.live_streams = (response.data.total_active_streams || 0) - response.data.frozen_streams;
      
      return { ...response.data, mode: 'basic' };
    } catch (error) { 
      console.error("❌ Error getting basic streaming stats:", error);
      throw error;
    }
  }

  cleanup = async () => { 
    try {
      console.log('🧹 Starting BasicStreamManager cleanup...');
      
      // Stop all basic streams
      const basicStreams = Array.from(this.detectionService.currentStreams.entries())
        .filter(([_, stream]) => stream.type && stream.type.startsWith('basic'));
      
      const stopPromises = basicStreams.map(([cameraId, _]) => this.stopBasicStream(cameraId));
      await Promise.allSettled(stopPromises);
      
      // Unfreeze all frozen streams
      const frozenStreams = this.detectionService.getFrozenStreams();
      for (const frozenStream of frozenStreams) {
        try {
          await this.unfreezeStream(frozenStream.cameraId);
        } catch (error) {
          console.warn(`⚠️ Error unfreezing stream ${frozenStream.cameraId}:`, error.message);
        }
      }
      
      console.log('✅ BasicStreamManager cleanup completed');
    } catch (error) {
      console.error("❌ Error during BasicStreamManager cleanup:", error);
    }
  };
}