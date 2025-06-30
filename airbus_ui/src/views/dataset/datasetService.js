// services/datasetService.js
import api from "../../utils/UseAxios";

export const datasetService = {
  // Enhanced method to get datasets with filters and pagination
  getAllDatasetsWithFilters: async (params = {}) => {
    try {
      const queryParams = new URLSearchParams();
      
      // Add all parameters that have values
      Object.entries(params).forEach(([key, value]) => {
        if (value !== '' && value !== null && value !== undefined) {
          queryParams.append(key, value);
        }
      });

      const response = await api.get(
        `/api/artifact_keeper/datasetManager/datasets/enhanced?${queryParams.toString()}`
      );
      return response.data;
    } catch (error) {
      console.error("Error fetching filtered datasets:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get dataset statistics
  getDatasetStatistics: async () => {
    try {
      const response = await api.get("/api/artifact_keeper/datasetManager/statistics");
      return response.data;
    } catch (error) {
      console.error("Error fetching dataset statistics:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get available groups for filtering
  getAvailableGroups: async () => {
    try {
      const response = await api.get("/api/artifact_keeper/datasetManager/groups");
      return response.data.groups || [];
    } catch (error) {
      console.error("Error fetching available groups:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Export dataset report
  exportDatasetReport: async (format = 'json') => {
    try {
      const response = await api.get(
        `/api/artifact_keeper/datasetManager/export?format_type=${format}`,
        {
          responseType: format === 'csv' ? 'blob' : 'json'
        }
      );
      return response.data;
    } catch (error) {
      console.error("Error exporting dataset report:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Bulk update multiple pieces
  bulkUpdatePieces: async (pieceIds, updates) => {
    try {
      const response = await api.patch("/api/artifact_keeper/datasetManager/pieces/bulk-update", {
        piece_ids: pieceIds,
        updates: updates
      });
      return response.data;
    } catch (error) {
      console.error("Error bulk updating pieces:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get piece labels by group
  getPieceLabelsByGroup: async (groupLabel) => {
    try {
      const response = await api.get(`/api/artifact_keeper/datasetManager/pieces/group/${groupLabel}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching piece labels by group:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get annotations for a specific piece
  getPieceAnnotations: async (pieceLabel) => {
    try {
      const response = await api.get(`/api/artifact_keeper/datasetManager/pieces/${pieceLabel}/annotations`);
      return response.data;
    } catch (error) {
      console.error("Error fetching piece annotations:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete a specific piece by label
  deletePieceByLabel: async (pieceLabel) => {
    try {
      const response = await api.delete(`/api/artifact_keeper/datasetManager/pieces/${pieceLabel}`);
      console.log(response.data.message || "Piece deleted successfully");
      return response.data;
    } catch (error) {
      console.error("Error deleting piece:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete all pieces
  deleteAllPieces: async () => {
    try {
      const response = await api.delete("/api/artifact_keeper/datasetManager/pieces");
      console.log(response.data.message || "All pieces deleted successfully");
      return response.data;
    } catch (error) {
      console.error("Error deleting all pieces:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete a specific annotation
  deleteAnnotation: async (annotationId) => {
    try {
      const response = await api.delete(`/api/artifact_keeper/datasetManager/annotations/${annotationId}`);
      console.log(response.data.message || "Annotation deleted successfully");
      return response.data;
    } catch (error) {
      console.error("Error deleting annotation:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Legacy method to get all datasets (for backward compatibility)
  getAllDatasets: async () => {
    try {
      const response = await api.get("/api/artifact_keeper/datasetManager/datasets");
      return response.data || {};
    } catch (error) {
      console.error("Error fetching datasets:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // === TRAINING SERVICES ===
  
  // Train a specific piece model
  trainPieceModel: async (pieceLabels) => {
    try {
      const response = await api.post("/api/training/training/train", {
        piece_labels: Array.isArray(pieceLabels) ? pieceLabels : [pieceLabels]
      });
      console.log(response.data.message || "Training started successfully");
      return response.data;
    } catch (error) {
      console.error("Error starting training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Train multiple pieces
  trainMultiplePieces: async (pieceLabels) => {
    try {
      const response = await api.post("/api/training/training/train", {
        piece_labels: pieceLabels
      });
      console.log(response.data.message || "Training started successfully");
      return response.data;
    } catch (error) {
      console.error("Error starting training for multiple pieces:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Train all non-trained pieces
  trainAllPieces: async () => {
    try {
      // First get all datasets to find non-trained pieces
      const datasetsResponse = await api.get("/api/artifact_keeper/datasetManager/datasets");
      const datasets = Object.values(datasetsResponse.data || {});
      
      // Filter non-trained pieces
      const nonTrainedPieces = datasets
        .filter(piece => !piece.is_yolo_trained)
        .map(piece => piece.label);
      
      if (nonTrainedPieces.length === 0) {
        throw new Error("No pieces available for training");
      }

      const response = await api.post("/api/training/training/train", {
        piece_labels: nonTrainedPieces
      });
      console.log(response.data.message || "Training started for all pieces");
      return response.data;
    } catch (error) {
      console.error("Error starting training for all pieces:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Stop training process
  stopTraining: async () => {
    try {
      const response = await api.post("/api/training/training/stop_training");
      console.log(response.data.message || "Training stopped successfully");
      return response.data;
    } catch (error) {
      console.error("Error stopping training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get training status/progress (if you have an endpoint for this)
  getTrainingStatus: async () => {
    try {
      const response = await api.get("/api/training/training/status");
      return response.data;
    } catch (error) {
      console.error("Error fetching training status:", error.response?.data?.detail || error.message);
      throw error;
    }
  }
};