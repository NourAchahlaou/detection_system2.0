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
      await api.post("/api/artifact_keeper/captureImage/cleanup-temp-photos");
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
      const response = await api.get(`/api/artifact_keeper/captureImage/capture_images/${pieceLabel}`, {
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

  // Get temporary images for a piece
  getTempImages: async (pieceLabel) => {
    try {
      const response = await api.get(`/api/artifact_keeper/captureImage/temp-photos/${pieceLabel}`);
      return response.data || [];
    } catch (error) {
      console.error("Error fetching temp images:", error.response?.data?.detail || error.message);
      return [];
    }
  },

  // Get temporary image blob by name
  getTempImageBlob: async (imageName) => {
    try {
      const response = await api.get(`/api/artifact_keeper/captureImage/temp-image/${imageName}`, {
        responseType: 'blob'
      });
      
      if (response.data instanceof Blob) {
        return URL.createObjectURL(response.data);
      }
      return null;
    } catch (error) {
      console.error("Error fetching temp image blob:", error.response?.data?.detail || error.message);
      return null;
    }
  },

  // Delete temporary image - NEW FUNCTION
  deleteTempImage: async (pieceLabel, imageName) => {
    try {
      const response = await api.delete(`/api/artifact_keeper/captureImage/temp-image/${pieceLabel}/${imageName}`);
      
      if (response.status === 200) {
        console.log(`Successfully deleted temp image: ${imageName}`);
        return true;
      }
      return false;
    } catch (error) {
      console.error("Error deleting temp image:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Get images by label (for saved images) - Fixed the 'this' context issue
  getImagesByLabel: async (targetLabel) => {
    try {
      // First try to get temporary images - using cameraService instead of 'this'
      const tempImages = await cameraService.getTempImages(targetLabel);
      
      // Convert temp images to proper format with blob URLs
      const processedTempImages = await Promise.all(
        tempImages.map(async (img) => {
          const blobUrl = await cameraService.getTempImageBlob(img.image_name);
          return {
            ...img,
            url: blobUrl,
            src: blobUrl,
            isTemporary: true
          };
        })
      );

      // You can also fetch saved images from database here if needed
      // const savedImages = await cameraService.getSavedImages(targetLabel);
      
      return processedTempImages;
    } catch (error) {
      console.error("Error fetching images by label:", error);
      return [];
    }
  },

  // Save captured images to database
  saveImagesToDatabase: async (pieceLabel) => {
    try {
      const response = await api.post("/api/artifact_keeper/captureImage/save-images", {
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
      await api.post("/api/artifact_keeper/captureImage/cleanup-temp-photos");
      return true;
    } catch (error) {
      console.error("Error during cleanup:", error);
      return false;
    }
  }
};