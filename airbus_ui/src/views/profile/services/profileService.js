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
   * Backend expects: name, email, airbus_id, role, password (all optional)
   */
  async updateProfileTabInfo(data) {
    try {
      // Transform frontend data to match backend schema
      const backendData = {
        name: data.full_name || data.name, // Single name field
        email: data.email,
        airbus_id: data.airbus_id ? parseInt(data.airbus_id) : undefined,
        role: data.role,
        password: data.password // Include password if provided
      };

      // Remove undefined fields
      Object.keys(backendData).forEach(key => {
        if (backendData[key] === undefined) {
          delete backendData[key];
        }
      });

      const response = await api.put('/api/users/profile-tab/update', backendData);
      return response.data;
    } catch (error) {
      console.error('Error updating profile tab info:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Get all shifts for the current user
   */
  async getAllShifts() {
    try {
      const response = await api.get('/api/users/profile-tab/shifts');
      return response.data;
    } catch (error) {
      console.error('Error fetching all shifts:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Update user shifts with support for individual shift modifications
   * @param {Array} shiftsData - Array of shift update objects with format:
   * [{
   *   day_of_week: "MONDAY",
   *   start_time: "09:00",
   *   end_time: "17:00",
   *   action: "update" // "create", "update", or "delete"
   * }]
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
   * @param {Object} bulkData - Object with format:
   * {
   *   start_time: "09:00",
   *   end_time: "17:00",
   *   days: ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]
   * }
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
   * Delete a specific shift for the current user
   * @param {string} dayOfWeek - Day of the week (MONDAY, TUESDAY, etc.)
   */
  async deleteShift(dayOfWeek) {
    try {
      const response = await api.delete(`/api/users/profile-tab/shifts/${dayOfWeek.toUpperCase()}`);
      return response.data;
    } catch (error) {
      console.error(`Error deleting shift for ${dayOfWeek}:`, error);
      throw this.handleError(error);
    }
  }

  /**
   * Get detailed user shifts organized by day with additional metadata
   * Returns enhanced data including average, max, and min daily hours
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
   * Returns basic work summary without the enhanced daily metrics
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
   * @param {string} fieldName - Name of the field to update
   * @param {string} fieldValue - New value for the field
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
      const response = await api.delete('/api/users/profile-tab/delete');
      return response.data;
    } catch (error) {
      console.error('Error deleting user profile:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Utility method to format shift data for API consumption
   * @param {Object} shiftData - Raw shift data
   * @param {string} action - Action to perform ("create", "update", "delete")
   * @returns {Object} Formatted shift data
   */
  formatShiftData(shiftData, action = 'update') {
    return {
      day_of_week: shiftData.dayOfWeek || shiftData.day_of_week,
      start_time: shiftData.startTime || shiftData.start_time,
      end_time: shiftData.endTime || shiftData.end_time,
      action: action
    };
  }

  /**
   * Utility method to format bulk shift data for API consumption
   * @param {string} startTime - Start time in HH:MM format
   * @param {string} endTime - End time in HH:MM format
   * @param {Array<string>} days - Array of day names
   * @returns {Object} Formatted bulk shift data
   */
  formatBulkShiftData(startTime, endTime, days) {
    return {
      start_time: startTime,
      end_time: endTime,
      days: days.map(day => day.toUpperCase())
    };
  }

  /**
   * Utility method to validate time format
   * @param {string} time - Time string to validate (HH:MM format)
   * @returns {boolean} True if valid, false otherwise
   */
  validateTimeFormat(time) {
    const timeRegex = /^([01]?[0-9]|2[0-3]):[0-5][0-9]$/;
    return timeRegex.test(time);
  }

  /**
   * Utility method to validate day of week
   * @param {string} day - Day string to validate
   * @returns {boolean} True if valid, false otherwise
   */
  validateDayOfWeek(day) {
    const validDays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'];
    return validDays.includes(day.toUpperCase());
  }

  /**
   * Process detailed shifts response to extract useful metrics
   * @param {Object} detailedShifts - Response from getDetailedShifts()
   * @returns {Object} Processed shift analytics
   */
  processShiftAnalytics(detailedShifts) {
    if (!detailedShifts || !detailedShifts.shifts_by_day) {
      return null;
    }

    const shifts = detailedShifts.shifts_by_day;
    const workingDays = Object.keys(shifts);

    return {
      totalWorkingDays: workingDays.length,
      totalWeeklyHours: detailedShifts.total_weekly_hours || 0,
      totalWeeklyMinutes: detailedShifts.total_weekly_minutes || 0,
      weeklyHoursDisplay: detailedShifts.total_weekly_display || '0h 0m',
      averageDailyHours: detailedShifts.average_daily_hours || '0h 0m',
      maxDailyHours: detailedShifts.max_daily_hours || '0h 0m',
      minDailyHours: detailedShifts.min_daily_hours || '0h 0m',
      workingDays: workingDays,
      shifts: shifts,
      // Helper methods
      isFullTime: () => (detailedShifts.total_weekly_hours || 0) >= 40,
      hasWeekendWork: () => workingDays.some(day => ['Saturday', 'Sunday'].includes(day)),
      getShiftsByDay: (day) => shifts[day] || null
    };
  }

  /**
   * Handle API errors consistently
   * @param {Error} error - The error object from the API call
   * @returns {Error} Processed error with user-friendly message
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
        case 422:
          return new Error('Invalid data format. Please check your input.');
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