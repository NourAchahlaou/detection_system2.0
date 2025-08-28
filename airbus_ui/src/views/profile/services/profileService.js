// services/profileService.js
import api from '../../../utils/UseAxios'; // Adjust path based on your project structure

class ProfileService {
  /**
   * Get complete user profile information for the profile tab display
   */
  async getProfileTabInfo() {
    try {
      const response = await api.get('/api/users/profile-tab/info');
      return response.data;
    } catch (error) {
      console.error('Error fetching profile tab info:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Update user profile information from the profile tab
   */
  async updateProfileTabInfo(data) {
    try {
      const response = await api.put('/api/users/profile-tab/update', data);
      return response.data;
    } catch (error) {
      console.error('Error updating profile tab info:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Update user shifts with support for individual shift modifications
   */
  async updateUserShifts(shiftsData) {
    try {
      const response = await api.put('/api/users/profile-tab/shifts/update', shiftsData);
      return response.data;
    } catch (error) {
      console.error('Error updating user shifts:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Bulk update regular shifts for multiple days with the same time
   */
  async bulkUpdateRegularShifts(bulkData) {
    try {
      const response = await api.put('/api/users/profile-tab/shifts/bulk-update', bulkData);
      return response.data;
    } catch (error) {
      console.error('Error bulk updating regular shifts:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Get detailed user shifts organized by day with additional metadata
   */
  async getDetailedShifts() {
    try {
      const response = await api.get('/api/users/profile-tab/shifts/detailed');
      return response.data;
    } catch (error) {
      console.error('Error fetching detailed shifts:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Get user work summary including shifts and total hours
   */
  async getWorkSummary() {
    try {
      const response = await api.get('/api/users/profile-tab/work-summary');
      return response.data;
    } catch (error) {
      console.error('Error fetching work summary:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Update a specific profile field
   */
  async updateProfileField(fieldName, fieldValue) {
    try {
      const response = await api.patch('/api/users/profile-tab/update-field', null, {
        params: {
          field_name: fieldName,
          field_value: fieldValue
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error updating profile field:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Delete user profile (soft delete - sets user as inactive)
   */
  async deleteUserProfile() {
    try {
      const response = await api.delete(' /api/users/profile-tab/delete');
      return response.data;
    } catch (error) {
      console.error('Error deleting user profile:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Handle API errors consistently
   */
  handleError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          return new Error('Authentication required. Please log in again.');
        case 403:
          return new Error('You do not have permission to perform this action.');
        case 404:
          return new Error('Profile information not found.');
        case 400:
          return new Error(data.detail || 'Invalid request data.');
        case 500:
          return new Error('Server error. Please try again later.');
        default:
          return new Error(data.detail || 'An unexpected error occurred.');
      }
    } else if (error.request) {
      // Network error
      return new Error('Network error. Please check your connection and try again.');
    } else {
      // Other error
      return new Error('An unexpected error occurred.');
    }
  }
}

// Export a singleton instance
const profileService = new ProfileService();
export default profileService;