// basicDetectionService.js - Service for basic (on-demand) detection
import api from "../../utils/UseAxios";

class BasicDetectionService {
  constructor() {
    this.isInitialized = false;
    this.isInitializing = false;
    this.lastDetectionResult = null;
    this.detectionStats = {
      totalDetections: 0,
      successfulDetections: 0,
      avgProcessingTime: 0,
      lastDetectionTime: null
    };
  }

  /**
   * Initialize the basic detection processor
   */
  async initialize() {
    if (this.isInitialized) {
      return { success: true, message: 'Already initialized' };
    }

    if (this.isInitializing) {
      throw new Error('Initialization already in progress');
    }

    this.isInitializing = true;

    try {
      console.log("🚀 Initializing basic detection processor...");
      
      const response = await api.post('/api/detection/basic/initialize');
      
      if (response.data.success) {
        this.isInitialized = true;
        console.log("✅ Basic detection processor initialized successfully");
        
        return {
          success: true,
          message: response.data.message,
          device: response.data.device
        };
      } else {
        throw new Error(response.data.message || 'Initialization failed');
      }
      
    } catch (error) {
      console.error("❌ Error initializing basic detection processor:", error);
      throw new Error(`Failed to initialize: ${error.response?.data?.detail || error.message}`);
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Check health of basic detection service
   */
  async checkHealth() {
    try {
      const response = await api.get('/api/detection/basic/health');
      return {
        healthy: response.data.status === 'healthy',
        details: response.data
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message
      };
    }
  }

  /**
   * Perform detection on a single frame from camera
   */
  async detectSingleFrame(cameraId, targetLabel, options = {}) {
    try {
      if (!this.isInitialized) {
        await this.initialize();
      }

      const { quality = 85 } = options;

      console.log(`🎯 Performing basic detection on camera ${cameraId} for target: ${targetLabel}`);

      const startTime = Date.now();
      
      const response = await api.post(`/api/detection/basic/detect/${cameraId}`, {
        target_label: targetLabel,
        quality: quality
      });

      const processingTime = Date.now() - startTime;

      if (response.data.success) {
        const detectionData = response.data.data;
        
        // Update stats
        this.detectionStats.totalDetections++;
        this.detectionStats.lastDetectionTime = new Date().toISOString();
        
        if (detectionData.detected_target) {
          this.detectionStats.successfulDetections++;
        }
        
        // Update average processing time
        this.detectionStats.avgProcessingTime = 
          (this.detectionStats.avgProcessingTime * (this.detectionStats.totalDetections - 1) + processingTime) / 
          this.detectionStats.totalDetections;

        this.lastDetectionResult = {
          ...detectionData,
          clientProcessingTime: processingTime
        };

        console.log(`✅ Basic detection completed in ${processingTime}ms:`, {
          detected: detectionData.detected_target,
          confidence: detectionData.confidence,
          processingTime: detectionData.processing_time_ms
        });

        return {
          success: true,
          data: this.lastDetectionResult,
          stats: this.getStats()
        };
      } else {
        throw new Error(response.data.message || 'Detection failed');
      }

    } catch (error) {
      console.error("❌ Error in basic detection:", error);
      
      // Update stats for failed detection
      this.detectionStats.totalDetections++;
      
      throw new Error(`Detection failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  /**
   * Test camera connectivity by capturing a frame
   */
  async testCameraFrame(cameraId) {
    try {
      console.log(`📸 Testing frame capture from camera ${cameraId}`);
      
      const response = await api.post(`/api/detection/basic/test/${cameraId}`);
      
      if (response.data.success) {
        console.log(`✅ Successfully captured test frame from camera ${cameraId}`);
        return {
          success: true,
          frameShape: response.data.frame_shape,
          frameData: response.data.frame_data
        };
      } else {
        throw new Error(response.data.message || 'Frame capture failed');
      }
      
    } catch (error) {
      console.error(`❌ Error testing camera ${cameraId}:`, error);
      throw new Error(`Camera test failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  /**
   * Perform batch detection on multiple cameras/targets
   */
  async detectBatch(detections) {
    try {
      if (!this.isInitialized) {
        await this.initialize();
      }

      console.log(`🎯 Performing batch detection on ${detections.length} items`);

      const response = await api.post('/api/detection/basic/detect/batch', {
        detections: detections
      });

      if (response.data.success) {
        console.log(`✅ Batch detection completed: ${response.data.successful}/${response.data.total_processed} successful`);
        return response.data;
      } else {
        throw new Error('Batch detection failed');
      }

    } catch (error) {
      console.error("❌ Error in batch detection:", error);
      throw new Error(`Batch detection failed: ${error.response?.data?.detail || error.message}`);
    }
  }

  /**
   * Get detection statistics
   */
  getStats() {
    return {
      ...this.detectionStats,
      isInitialized: this.isInitialized,
      isInitializing: this.isInitializing,
      lastResult: this.lastDetectionResult ? {
        detected: this.lastDetectionResult.detected_target,
        confidence: this.lastDetectionResult.confidence,
        processingTime: this.lastDetectionResult.processing_time_ms,
        timestamp: this.lastDetectionResult.timestamp
      } : null
    };
  }

  /**
   * Get service status
   */
  async getServiceStats() {
    try {
      const response = await api.get('/api/detection/basic/stats');
      return response.data.stats;
    } catch (error) {
      console.error("Error getting service stats:", error);
      return null;
    }
  }

  /**
   * Reset detection statistics
   */
  resetStats() {
    this.detectionStats = {
      totalDetections: 0,
      successfulDetections: 0,
      avgProcessingTime: 0,
      lastDetectionTime: null
    };
    this.lastDetectionResult = null;
    console.log("🔄 Basic detection stats reset");
  }

  /**
   * Get success rate
   */
  getSuccessRate() {
    if (this.detectionStats.totalDetections === 0) return 0;
    return (this.detectionStats.successfulDetections / this.detectionStats.totalDetections) * 100;
  }

  /**
   * Check if service is ready for detection
   */
  isReady() {
    return this.isInitialized && !this.isInitializing;
  }

  /**
   * Force re-initialize the service
   */
  async reinitialize() {
    this.isInitialized = false;
    this.isInitializing = false;
    return await this.initialize();
  }
}

// Export singleton instance
export const basicDetectionService = new BasicDetectionService();