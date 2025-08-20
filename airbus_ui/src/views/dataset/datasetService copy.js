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

  // === ENHANCED TRAINING SERVICES ===
  
  // Train a specific piece model
  trainPieceModel: async (pieceLabel) => {
    try {
      const response = await api.post("/api/training/training/train", {
        piece_labels: Array.isArray(pieceLabel) ? pieceLabel : [pieceLabel]
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
        piece_labels: Array.isArray(pieceLabels) ? pieceLabels : [pieceLabels]
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
      const datasetsResponse = await api.get("/api/artifact_keeper/datasetManager/datasets");
      const datasets = Object.values(datasetsResponse.data || {});
      
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
      const response = await api.post("/api/training/training/stop");
      console.log(response.data.message || "Training stopped successfully");
      return response.data;
    } catch (error) {
      console.error("Error stopping training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get training status/progress with logs option
  getTrainingStatus: async (includeLogs = false) => {
    try {
      const response = await api.get(`/api/training/training/status?include_logs=${includeLogs}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching training status:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Check if training is currently active
  isTrainingActive: async () => {
    try {
      const response = await api.get("/api/training/training/status");
      return response.data?.data?.is_training || false;
    } catch (error) {
      console.error("Error checking training status:", error.response?.data?.detail || error.message);
      return false;
    }
  },

  // Get training logs with filtering
  getTrainingLogs: async (params = {}) => {
    try {
      const queryParams = new URLSearchParams();
      
      // Add parameters
      if (params.session_id) queryParams.append('session_id', params.session_id);
      if (params.limit) queryParams.append('limit', params.limit);
      if (params.level) queryParams.append('level', params.level);
      
      const response = await api.get(`/api/training/training/logs?${queryParams.toString()}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching training logs:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Update training progress
  updateTrainingProgress: async (progressData) => {
    try {
      const response = await api.put("/api/training/training/progress", progressData);
      return response.data;
    } catch (error) {
      console.error("Error updating training progress:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Health check for training service
  checkTrainingHealth: async () => {
    try {
      const response = await api.get("/api/training/training/health");
      return response.data;
    } catch (error) {
      console.error("Error checking training health:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // === NEW TRAINING SESSION MANAGEMENT ===

  // Get all training sessions with pagination and filtering
  getTrainingSessions: async (params = {}) => {
    try {
      const queryParams = new URLSearchParams();
      
      if (params.limit) queryParams.append('limit', params.limit);
      if (params.offset) queryParams.append('offset', params.offset);
      if (params.status_filter) queryParams.append('status_filter', params.status_filter);
      
      const response = await api.get(`/api/training/training/sessions?${queryParams.toString()}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching training sessions:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get specific training session details
  getTrainingSessionDetails: async (sessionId, includeLogs = true) => {
    try {
      const response = await api.get(`/api/training/training/session/${sessionId}?include_logs=${includeLogs}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching training session details:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete a training session
  deleteTrainingSession: async (sessionId, force = false) => {
    try {
      const response = await api.delete(`/api/training/training/session/${sessionId}?force=${force}`);
      console.log(response.data.message || "Training session deleted successfully");
      return response.data;
    } catch (error) {
      console.error("Error deleting training session:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get resumable training sessions
  getResumableSessions: async () => {
    try {
      const response = await api.get("/api/training/training/resumable");
      return response.data;
    } catch (error) {
      console.error("Error fetching resumable sessions:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Resume a specific training session
  resumeTrainingSession: async (sessionId) => {
    try {
      const response = await api.post(`/api/training/training/resume/${sessionId}`);
      console.log(response.data.message || "Training session resumed successfully");
      return response.data;
    } catch (error) {
      console.error("Error resuming training session:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Pause current training
  pauseTraining: async () => {
    try {
      const response = await api.post("/api/training/training/pause");
      console.log(response.data.message || "Training paused successfully");
      return response.data;
    } catch (error) {
      console.error("Error pausing training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get training statistics and analytics
  getTrainingStatistics: async () => {
    try {
      const response = await api.get("/api/training/training/statistics");
      return response.data;
    } catch (error) {
      console.error("Error fetching training statistics:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get session metrics for a specific session
  getSessionMetrics: async (sessionId) => {
    try {
      const response = await api.get(`/api/training/training/metrics/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching session metrics:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // === BATCH TRAINING OPERATIONS ===

  // Start batch training for multiple piece groups
  batchTrainMultiple: async (pieceGroups, sequential = false) => {
    try {
      const response = await api.post(`/api/training/training/batch/train?sequential=${sequential}`, {
        piece_groups: pieceGroups
      });
      console.log(response.data.message || "Batch training started successfully");
      return response.data;
    } catch (error) {
      console.error("Error starting batch training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // === CLEANUP AND EXPORT OPERATIONS ===

  // Cleanup old training sessions
  cleanupOldSessions: async (params = {}) => {
    try {
      const queryParams = new URLSearchParams();
      
      if (params.older_than_days) queryParams.append('older_than_days', params.older_than_days);
      if (params.keep_completed !== undefined) queryParams.append('keep_completed', params.keep_completed);
      if (params.dry_run !== undefined) queryParams.append('dry_run', params.dry_run);
      
      const response = await api.delete(`/api/training/training/cleanup?${queryParams.toString()}`);
      return response.data;
    } catch (error) {
      console.error("Error cleaning up old sessions:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Export training report
  exportTrainingReport: async (params = {}) => {
    try {
      const queryParams = new URLSearchParams();
      
      if (params.format_type) queryParams.append('format_type', params.format_type);
      if (params.include_logs !== undefined) queryParams.append('include_logs', params.include_logs);
      if (params.session_ids) {
        params.session_ids.forEach(id => queryParams.append('session_ids', id));
      }
      
      const response = await api.get(`/api/training/training/export/report?${queryParams.toString()}`);
      return response.data;
    } catch (error) {
      console.error("Error exporting training report:", error.response?.data?.detail || error.message);
      throw error;
    }
  }
  
};