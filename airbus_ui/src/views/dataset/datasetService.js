// services/datasetService.js
import api from "../../utils/UseAxios";

export const datasetService = {
  // Get all datasets
  getAllDatasets: async () => {
    try {
      const response = await api.get("/api/artifact_keeper/datasetManager/datasets");
      return response.data || {};
    } catch (error) {
      console.error("Error fetching datasets:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete a specific piece by label
  deletePieceByLabel: async (pieceLabel) => {
    try {
      const response = await api.delete(`/api/artifact_keeper/datasetManager/delete_piece/${pieceLabel}`);
      console.log(response.data.message || "Piece deleted successfully");
      return true;
    } catch (error) {
      console.error("Error deleting piece:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Delete all pieces
  deleteAllPieces: async () => {
    try {
      const response = await api.delete("/api/artifact_keeper/datasetManager/delete_all_pieces");
      console.log(response.data.message || "All pieces deleted successfully");
      return true;
    } catch (error) {
      console.error("Error deleting all pieces:", error.response?.data?.detail || error.message);
      throw error;
    }
  },

  // Train a model for a specific piece
  trainModel: async (pieceLabel) => {
    try {
      const response = await api.post(`/api/artifact_keeper/detection/train/${pieceLabel}`);
      console.log(response.data.message || "Training started successfully");
      return response.data;
    } catch (error) {
      console.error("Error starting training:", error.response?.data?.detail || error.message);
      throw error;
    }
  }
};