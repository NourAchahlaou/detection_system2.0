// services/cameraService.js
import api from "../../utils/UseAxios";

export const cameraService = {
  // Start camera with given ID
  startCamera: async (cameraId) => {
    try {
      const numericCameraId = parseInt(cameraId);
      if (isNaN(numericCameraId)) {
        throw new Error("Invalid camera ID: must be a number");
      }

      const response = await api.post("/api/artifact_keeper/camera/start", { 
        camera_id: numericCameraId
      });
      
      console.log(response.data.message || "Camera started successfully");
      return true;
    } catch (error) {
      console.error("Error starting camera:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Stop camera and cleanup
  stopCamera: async () => {
    try {
      await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
      const response = await api.post("/api/artifact_keeper/camera/stop");
      
      console.log("Camera stopped and temporary photos cleaned up.");
      return true;
    } catch (error) {
      console.error("Error stopping camera:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Capture images for a piece
  captureImages: async (pieceLabel) => {
    try {
      const response = await api.get(`/api/artifact_keeper/camera/capture_images/${pieceLabel}`, {
        responseType: 'blob'
      });
      
      if (response.data instanceof Blob) {
        const imageUrl = URL.createObjectURL(response.data);
        return imageUrl;
      } else {
        console.error("Response is not a blob:", response.data);
        return null;
      }
    } catch (error) {
      console.error("Error capturing images:", error.response?.data?.detail || error.message);
      return null;
    }
  },

  // Save captured images to database
  saveImagesToDatabase: async (pieceLabel) => {
    try {
      const response = await api.post("/api/artifact_keeper/camera/save-images", {
        piece_label: pieceLabel
      });

      console.log(response.data.message || "Images saved successfully");
      return true;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || "Error saving images";
      console.error(errorMessage);
      return false;
    }
  },

  // Get all available cameras
  getAllCameras: async () => {
    try {
      const response = await api.get('/api/artifact_keeper/camera/get_allcameras');
      console.log("Cameras received:", response.data);
      return response.data;
    } catch (error) {
      console.error('Error fetching camera data:', error);
      return [];
    }
  },

  // Cleanup temporary photos
  cleanupTempPhotos: async () => {
    try {
      await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
      return true;
    } catch (error) {
      console.error("Error during cleanup:", error);
      return false;
    }
  }
};