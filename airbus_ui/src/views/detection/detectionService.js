// services/detectionService.js
import api from "../../utils/UseAxios";

export const detectionService = {
  // Load/reload the detection model
  loadModel: async () => {
    try {
      const response = await api.post("/api/detection/detection/model/reload");
      console.log("Model loaded successfully:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error loading model:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Check detection service health
  checkDetectionHealth: async () => {
    try {
      const response = await api.get("/api/detection/health");
      return response.data;
    } catch (error) {
      console.error("Error checking detection health:", error.response?.data?.detail || error.message);
      return null;
    }
  },

  // Start video streaming (via video streaming service)
  startVideoStream: async (cameraId) => {
    try {
      const numericCameraId = parseInt(cameraId);
      if (isNaN(numericCameraId)) {
        throw new Error("Invalid camera ID: must be a number");
      }

      const response = await api.post("/api/artifact_keeper/camera/start", {
        camera_id: numericCameraId
      });
      
      console.log("Video stream started:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error starting video stream:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Stop video streaming (via video streaming service)
  stopVideoStream: async (cameraId) => {
    try {
      const response = await api.post(`/api/video_streaming/video/stop/${cameraId}`);
      console.log("Video stream stopped:", response.data.message);
      return response.data;
    } catch (error) {
      console.error("Error stopping video stream:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get video feed URL (from video streaming service)
  getVideoFeedUrl: (cameraId) => {
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }

    return `/api/video_streaming/video/stream/${numericCameraId}`;
  },

  // Get video stream status
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
      return null;
    }
  },

  // Get all video streams status
  getAllVideoStreamsStatus: async () => {
    try {
      const response = await api.get("/api/video_streaming/video/status");
      return response.data;
    } catch (error) {
      console.error("Error getting all video streams status:", error.response?.data?.detail || error.message);
      return null;
    }
  },

  // Check video streaming service health
  checkVideoStreamingHealth: async () => {
    try {
      const response = await api.get("/api/video_streaming/video/health");
      return response.data;
    } catch (error) {
      console.error("Error checking video streaming health:", error.response?.data?.detail || error.message);
      return null;
    }
  },

  // Get WebSocket URL for real-time video streaming
  getWebSocketVideoUrl: (cameraId) => {
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }

    // Determine WebSocket protocol based on current protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    return `${protocol}//${host}/api/video_streaming/video/ws/${numericCameraId}`;
  },

  // Legacy method for backward compatibility - kept for artifact_keeper integration
  startDetectionFeed: async (cameraId, targetLabel) => {
    try {
      const numericCameraId = parseInt(cameraId);
      if (isNaN(numericCameraId)) {
        throw new Error("Invalid camera ID: must be a number");
      }

      // Start camera via artifact keeper first
      const cameraResponse = await api.post("/api/artifact_keeper/camera/start", {
        camera_id: numericCameraId
      });
      
      console.log("Camera started for detection:", cameraResponse.data.message);
      
      // Return the video feed URL with target label as parameter
      return `/api/video_streaming/video/stream/${numericCameraId}?target_label=${encodeURIComponent(targetLabel)}`;
    } catch (error) {
      console.error("Error starting detection feed:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Legacy method for backward compatibility - kept for artifact_keeper integration
  stopDetectionFeed: async () => {
    try {
      // Stop camera via artifact keeper
      await api.post("/api/artifact_keeper/camera/stop");
      
      console.log("Detection feed stopped successfully");
      return true;
    } catch (error) {
      console.error("Error stopping detection feed:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Legacy method for backward compatibility
  checkCamera: async () => {
    console.warn("checkCamera is deprecated. Use checkVideoStreamingHealth instead.");
    return detectionService.checkVideoStreamingHealth();
  },

  // Comprehensive service health check
  checkAllServices: async () => {
    try {
      const [detectionHealth, videoHealth] = await Promise.all([
        detectionService.checkDetectionHealth(),
        detectionService.checkVideoStreamingHealth()
      ]);

      return {
        detection: detectionHealth,
        video_streaming: videoHealth,
        overall_status: (detectionHealth?.status === 'healthy' && videoHealth?.status === 'healthy') 
          ? 'healthy' 
          : 'degraded'
      };
    } catch (error) {
      console.error("Error checking all services:", error);
      return {
        detection: null,
        video_streaming: null,
        overall_status: 'unhealthy'
      };
    }
  }
};