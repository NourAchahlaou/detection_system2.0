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
  // deletePieceByLabel: async (pieceLabel) => {
  //   try {
  //     const response = await api.delete(`/api/artifact_keeper/datasetManager/pieces/${pieceLabel}`);
  //     console.log(response.data.message || "Piece deleted successfully");
  //     return response.data;
  //   } catch (error) {
  //     console.error("Error deleting piece:", error.response?.data?.detail || error.message);
  //     throw error;
  //   }
  // },
// FIXED: deleteBatchOfPieces in datasetService.js
deleteBatchOfPieces: async (pieceLabels) => {
  console.log("Deleting batch of pieces:", pieceLabels);
  
  try {
    // FIXED: Validate and clean input
    if (!Array.isArray(pieceLabels)) {
      throw new Error('pieceLabels must be an array');
    }
    
    // FIXED: Flatten any nested arrays and ensure we have strings
    const cleanLabels = pieceLabels.flat().filter(label => 
      typeof label === 'string' && label.trim() !== ''
    );
    
    if (cleanLabels.length === 0) {
      throw new Error('No valid piece labels provided');
    }
    
    console.log("Clean labels for deletion:", cleanLabels);
    
    const response = await api.delete(
      "/api/artifact_keeper/datasetManager/pieces/batch",
      {
        data: { piece_labels: cleanLabels }  // Send clean array of strings
      }
    );

    console.log(response.data.message || "Batch of pieces deleted successfully");
    return response.data;
  } catch (error) {
    console.error(
      "Error deleting batch of pieces:",
      error.response?.data?.detail || error.message
    );
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
  
  // Train a specific piece model - FIXED
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

  // Train multiple pieces - FIXED
// Fixed version of trainMultiplePieces in your datasetService
trainMultiplePieces: async (pieceLabels) => {
  console.log('=== datasetService.trainMultiplePieces START ===');
  console.log('Input:', pieceLabels);
  
  try {
    // Validate input
    if (!Array.isArray(pieceLabels) || pieceLabels.length === 0) {
      console.error('Invalid input to trainMultiplePieces:', pieceLabels);
      const errorResult = { success: false, error: 'Invalid piece labels: must be a non-empty array' };
      console.log('Returning error result:', errorResult);
      return errorResult; // ✅ EXPLICIT RETURN
    }
    
    const requestPayload = {
      piece_labels: pieceLabels
    };
    
    console.log('Making API request with payload:', requestPayload);
    console.log('API endpoint: /api/training/training/train');
    
    // Check if api object exists
    if (!api) {
      console.error('API object not available');
      const errorResult = { success: false, error: 'API client not available' };
      console.log('Returning error result:', errorResult);
      return errorResult; // ✅ EXPLICIT RETURN
    }
    
    console.log('API client available, making POST request...');
    
    const response = await api.post("/api/training/training/train", requestPayload);
    console.log('API response received:', response);
    console.log('Response data:', response.data);
    console.log('Response status:', response.status);
    
    // ✅ FIXED: More robust response handling
    if (response && response.status >= 200 && response.status < 300) {
      // Success response - wrap the API response data
      const successResult = {
        success: true,
        data: response.data,
        message: response.data?.message || 'Training started successfully',
        session_id: response.data?.session_id
      };
      console.log('Success! Returning success result:', successResult);
      return successResult; // ✅ EXPLICIT RETURN WITH WRAPPED DATA
    } else {
      // Unexpected response format
      console.error('Unexpected response format:', response);
      const errorResult = { 
        success: false, 
        error: 'Unexpected response format',
        response: response
      };
      console.log('Returning error result:', errorResult);
      return errorResult; // ✅ EXPLICIT RETURN
    }
  } catch (error) {
    console.error('Exception in trainMultiplePieces:', error);
    console.log('Error response data:', error.response?.data);
    console.log('Error status:', error.response?.status);
    
    const errorResult = {
      success: false,
      error: error.response?.data?.detail || error.message || 'Unknown error occurred',
      statusCode: error.response?.status,
      details: error.response?.data
    };
    console.log('Returning exception result:', errorResult);
    return errorResult; // ✅ EXPLICIT RETURN
  } finally {
    console.log('=== datasetService.trainMultiplePieces END ===');
  }
},

  // Train all non-trained pieces - FIXED
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
      const response = await api.post("/api/training/training/stop");
      console.log(response.data.message || "Training stopped successfully");
      return response.data;
    } catch (error) {
      console.error("Error stopping training:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Get training status/progress - ENHANCED
  getTrainingStatus: async () => {
    try {
      const response = await api.get("/api/training/training/status");
      return response.data;
    } catch (error) {
      console.error("Error fetching training status:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // NEW: Check if training is currently active
  isTrainingActive: async () => {
    try {
      const response = await api.get("/api/training/training/status");
      return response.data?.data?.is_training || false;
    } catch (error) {
      console.error("Error checking training status:", error.response?.data?.detail || error.message);
      return false;
    }
  },
getTrainingSessions: async (params = {}) => {
  try {
    const queryParams = new URLSearchParams();
    
    // Add parameters
    Object.entries(params).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined) {
        queryParams.append(key, value);
      }
    });

    const response = await api.get(
      `/api/training/training/sessions?${queryParams.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching training sessions:", error.response?.data?.detail || error.message);
    throw error;
  }
},
  // NEW: Get training logs
  getTrainingLogs: async (limit = 50) => {
    try {
      const response = await api.get(`/api/training/training/logs?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching training logs:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // NEW: Update training progress (for real-time updates)
  updateTrainingProgress: async (progressData) => {
    try {
      const response = await api.put("/api/training/training/progress", progressData);
      return response.data;
    } catch (error) {
      console.error("Error updating training progress:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // NEW: Health check for training service
  checkTrainingHealth: async () => {
    try {
      const response = await api.get("/api/training/training/health");
      return response.data;
    } catch (error) {
      console.error("Error checking training health:", error.response?.data?.detail || error.message);
      throw error;
    }
  },
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

// Resume a training session
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

// Enhanced getTrainingStatus with logs option
getTrainingStatus: async (includeLogs = false) => {
  try {
    const response = await api.get(`/api/training/training/status?include_logs=${includeLogs}`);
    return response.data;
  } catch (error) {
    console.error("Error fetching training status:", error.response?.data?.detail || error.message);
    throw error;
  }
}
};