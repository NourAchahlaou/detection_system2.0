// services/detectionService.js
import api from "../../utils/UseAxios";

export const detectionService = {
  // === DETECTION SERVICE ENDPOINTS (Pure Detection - Port 8000) ===
  
  loadModel: async () => {
    try {
      const response = await api.get("/api/detection/detection/load_model");
      console.log("Model loaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error loading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  reloadModel: async () => {
    try {
      const response = await api.post("/api/detection/detection/model/reload");
      console.log("Model reloaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error reloading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  checkDetectionHealth: async () => {
    try {
      const response = await api.get("/api/detection/detection/health");
      return response.data;
    } catch (error) {
      console.error("Error checking detection health:", error.response?.data?.detail || error.message);
      return { status: 'unhealthy', model_loaded: false };
    }
  },

  getModelStatus: async () => {
    try {
      const response = await api.get("/api/detection/detection/model/status");
      return response.data;
    } catch (error) {
      console.error("Error getting model status:", error.response?.data?.detail || error.message);
      return { model_loaded: false };
    }
  },

  // === VIDEO STREAMING WITH DETECTION (Port 8001) ===
  
  // Main method: Get video stream URL with detection overlay
  getDetectionVideoFeedUrl: (cameraId, targetLabel = 'person') => {
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }
    
    const encodedLabel = encodeURIComponent(targetLabel);
    return `/api/video_streaming/video/stream_with_detection/${numericCameraId}?target_label=${encodedLabel}`;
  },

  // Fallback: Basic video stream (no detection)
  getBasicVideoFeedUrl: (cameraId) => {
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }
    return `/api/video_streaming/video/stream/${numericCameraId}`;
  },

  // === CAMERA LIFECYCLE MANAGEMENT ===
  
  startDetectionStream: async (cameraId, targetLabel = 'person') => {
    try {
      const numericCameraId = parseInt(cameraId);
      if (isNaN(numericCameraId)) {
        throw new Error("Invalid camera ID: must be a number");
      }

      // Ensure detection model is loaded first
      await detectionService.loadModel();

      // Start camera via artifact keeper (handles hardware initialization)
      const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
        camera_id: numericCameraId
      });
      
      console.log("Camera started for detection:", cameraResponse.data.message);
      
      return {
        success: true,
        videoUrl: detectionService.getDetectionVideoFeedUrl(cameraId, targetLabel),
        message: `Detection stream started for camera ${numericCameraId} with target: ${targetLabel}`,
        cameraResponse: cameraResponse.data
      };
    } catch (error) {
      console.error("Error starting detection stream:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  stopDetectionStream: async (cameraId) => {
    try {
      // Stop camera via artifact keeper (handles hardware cleanup)
      const response = await api.post("/api/artifact_keeper/camera/stop");
      
      console.log("Detection stream stopped successfully");
      return {
        success: true,
        message: `Detection stream stopped for camera ${cameraId}`,
        response: response.data
      };
    } catch (error) {
      console.error("Error stopping detection stream:", error.response?.data?.detail || error.message);
      return {
        success: false,
        message: "Failed to stop detection stream",
        error: error.response?.data?.detail || error.message
      };
    }
  },

  // === HEALTH AND STATUS CHECKS ===
  
  checkVideoStreamingHealth: async () => {
    try {
      const response = await api.get("/api/video_streaming/video/health");
      return response.data;
    } catch (error) {
      console.error("Error checking video streaming health:", error.response?.data?.detail || error.message);
      return { status: 'unhealthy' };
    }
  },

  getVideoStreamStatus: async (cameraId) => {
    try {
      const numericCameraId = parseInt(cameraId);
      if (isNaN(numericCameraId)) {
        throw new Error("Invalid camera ID: must be a number");
      }

      const response = await api.get(`/api/video_streaming/video/status/${numericCameraId}`);
      return response.data;
    } catch (error) {
      console.error("Error getting video stream status:", error.response?.data?.detail || error.message);
      return { active: false, error: error.message };
    }
  },

  getAllVideoStreamsStatus: async () => {
    try {
      const response = await api.get("/api/video_streaming/video/status");
      return response.data;
    } catch (error) {
      console.error("Error getting all video streams status:", error.response?.data?.detail || error.message);
      return { streams: [] };
    }
  },

  // === COMPREHENSIVE SERVICE HEALTH ===
  
  checkAllServices: async () => {
    try {
      const [detectionHealth, videoHealth, modelStatus] = await Promise.all([
        detectionService.checkDetectionHealth(),
        detectionService.checkVideoStreamingHealth(),
        detectionService.getModelStatus()
      ]);

      const overallStatus = (
        detectionHealth?.status === 'healthy' && 
        videoHealth?.status === 'healthy' &&
        modelStatus?.model_loaded === true
      ) ? 'healthy' : 'degraded';

      return {
        detection: {
          ...detectionHealth,
          model_loaded: modelStatus?.model_loaded || false
        },
        video_streaming: videoHealth,
        model: modelStatus,
        overall_status: overallStatus,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error("Error checking all services:", error);
      return {
        detection: { status: 'unhealthy', model_loaded: false },
        video_streaming: { status: 'unhealthy' },
        model: { model_loaded: false },
        overall_status: 'unhealthy',
        timestamp: new Date().toISOString()
      };
    }
  },

  // === UTILITY METHODS ===
  
  // Process a single frame directly via detection service
  processFrame: async (frameFile, targetLabel) => {
    try {
      const formData = new FormData();
      formData.append('frame', frameFile);
      formData.append('target_label', targetLabel);

      const response = await api.post('/api/detection/detection/process_frame', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 10000 // 10 second timeout for frame processing
      });

      return response.data;
    } catch (error) {
      console.error("Error processing frame:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // === LEGACY COMPATIBILITY METHODS ===
  
  // Keep these for backward compatibility with existing components
  startVideoStream: async (cameraId) => {
    console.warn("startVideoStream is deprecated. Use startDetectionStream for detection-enabled streams.");
    return detectionService.startDetectionStream(cameraId);
  },

  stopVideoStream: async (cameraId) => {
    console.warn("stopVideoStream is deprecated. Use stopDetectionStream instead.");
    return detectionService.stopDetectionStream(cameraId);
  },

  getVideoFeedUrl: (cameraId) => {
    console.warn("getVideoFeedUrl is deprecated. Use getDetectionVideoFeedUrl for detection or getBasicVideoFeedUrl for basic streaming.");
    return detectionService.getBasicVideoFeedUrl(cameraId);
  },

  startDetectionFeed: async (cameraId, targetLabel) => {
    console.warn("startDetectionFeed is deprecated. Use startDetectionStream instead.");
    const result = await detectionService.startDetectionStream(cameraId, targetLabel);
    return result.videoUrl;
  },

  stopDetectionFeed: async () => {
    console.warn("stopDetectionFeed is deprecated. Use stopDetectionStream instead.");
    const result = await detectionService.stopDetectionStream();
    return result.success;
  },

  checkCamera: async () => {
    console.warn("checkCamera is deprecated. Use checkVideoStreamingHealth instead.");
    return detectionService.checkVideoStreamingHealth();
  }
};