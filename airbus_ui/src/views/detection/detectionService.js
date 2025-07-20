// services/detectionService.js
import api from "../../utils/UseAxios";

export const detectionService = {
  // Load the detection model
  loadModel: async () => {
    try {
      await api.get("/api/detection/detection/load_model");
      console.log("Model loaded successfully");
      return true;
    } catch (error) {
      console.error("Error loading model:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Start detection video feed
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
      
      // Return the detection video feed URL
      return `/api/detection/detection/video_feed?camera_id=${numericCameraId}&target_label=${encodeURIComponent(targetLabel)}`;
    } catch (error) {
      console.error("Error starting detection feed:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Stop detection feed
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

  // Check camera status
  checkCamera: async () => {
    try {
      const response = await api.get("/api/detection/detection/check_camera");
      return response.data;
    } catch (error) {
      console.error("Error checking camera:", error.response?.data?.detail || error.message);
      return null;
    }
  }
};