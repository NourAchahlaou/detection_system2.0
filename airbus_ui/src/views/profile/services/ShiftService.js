import api from "../../../utils/UseAxios";

// Define the 4 states clearly
const ServiceStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY',
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class ShiftService {
  constructor() {
    this.state = ServiceStates.READY;
    this.eventListeners = new Map();
  }

  // ===================
  // SHIFT CRUD OPERATIONS
  // ===================

  async getAllUserShifts() {
    try {
      console.log('📅 Getting all user shifts...');
      
      const response = await api.get('/api/users/shifts/');
      
      if (response.data) {
        console.log(`✅ Retrieved ${response.data.length} shifts`);
        return {
          success: true,
          shifts: response.data,
          count: response.data.length,
          message: 'Shifts retrieved successfully'
        };
      } else {
        throw new Error('Failed to get user shifts');
      }
    } catch (error) {
      console.error('❌ Error getting user shifts:', error);
      throw new Error(`Failed to get shifts: ${error.response?.data?.detail || error.message}`);
    }
  }

  async getShiftById(shiftId) {
    try {
      console.log(`📅 Getting shift ${shiftId}...`);
      
      const response = await api.get(`/api/users/shifts/${shiftId}`);
      
      if (response.data) {
        console.log(`✅ Retrieved shift ${shiftId}: ${response.data.day_name} ${response.data.start_time}-${response.data.end_time}`);
        return {
          success: true,
          shift: response.data,
          message: 'Shift retrieved successfully'
        };
      } else {
        throw new Error('Failed to get shift');
      }
    } catch (error) {
      console.error(`❌ Error getting shift ${shiftId}:`, error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message;
      
      return {
        success: false,
        shift: null,
        message: errorMessage,
        error: errorMessage
      };
    }
  }

  async createShift(shiftData) {
    try {
      const { dayOfWeek, startTime, endTime } = shiftData;
      
      console.log(`📅 Creating shift: ${dayOfWeek} ${startTime}-${endTime}...`);
      
      const response = await api.post('/api/users/shifts/', {
        day_of_week: dayOfWeek,
        start_time: startTime,
        end_time: endTime
      });
      
      if (response.data) {
        console.log(`✅ Created shift ${response.data.id}: ${response.data.day_name} ${response.data.start_time}-${response.data.end_time}`);
        return {
          success: true,
          shift: response.data,
          shiftId: response.data.id,
          message: 'Shift created successfully'
        };
      } else {
        throw new Error('Failed to create shift');
      }
    } catch (error) {
      console.error('❌ Error creating shift:', error);
      throw new Error(`Failed to create shift: ${error.response?.data?.detail || error.message}`);
    }
  }

  async updateShift(shiftId, updateData) {
    try {
      console.log(`📅 Updating shift ${shiftId}...`);
      
      const requestBody = {};
      if (updateData.dayOfWeek) requestBody.day_of_week = updateData.dayOfWeek;
      if (updateData.startTime) requestBody.start_time = updateData.startTime;
      if (updateData.endTime) requestBody.end_time = updateData.endTime;
      
      const response = await api.put(`/api/users/shifts/${shiftId}`, requestBody);
      
      if (response.data) {
        console.log(`✅ Updated shift ${shiftId}: ${response.data.day_name} ${response.data.start_time}-${response.data.end_time}`);
        return {
          success: true,
          shift: response.data,
          message: 'Shift updated successfully'
        };
      } else {
        throw new Error('Failed to update shift');
      }
    } catch (error) {
      console.error(`❌ Error updating shift ${shiftId}:`, error);
      throw new Error(`Failed to update shift: ${error.response?.data?.detail || error.message}`);
    }
  }

  async deleteShift(shiftId) {
    try {
      console.log(`📅 Deleting shift ${shiftId}...`);
      
      const response = await api.delete(`/api/users/shifts/${shiftId}`);
      
      if (response.data) {
        console.log(`✅ Deleted shift ${shiftId}`);
        return {
          success: true,
          message: response.data.message || 'Shift deleted successfully'
        };
      } else {
        throw new Error('Failed to delete shift');
      }
    } catch (error) {
      console.error(`❌ Error deleting shift ${shiftId}:`, error);
      throw new Error(`Failed to delete shift: ${error.response?.data?.detail || error.message}`);
    }
  }

  async deleteAllUserShifts() {
    try {
      console.log('📅 Deleting all user shifts...');
      
      const response = await api.delete('/api/users/shifts/');
      
      if (response.data) {
        console.log(`✅ Deleted ${response.data.deleted_count} shifts`);
        return {
          success: true,
          deletedCount: response.data.deleted_count,
          message: response.data.message || 'All shifts deleted successfully'
        };
      } else {
        throw new Error('Failed to delete all shifts');
      }
    } catch (error) {
      console.error('❌ Error deleting all user shifts:', error);
      throw new Error(`Failed to delete all shifts: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // ADMIN OPERATIONS
  // ===================

  async getUserShiftsByUserId(userId) {
    try {
      console.log(`📅 Getting shifts for user ${userId}...`);
      
      const response = await api.get(`/api/users/shifts/user/${userId}`);
      
      if (response.data) {
        console.log(`✅ Retrieved ${response.data.length} shifts for user ${userId}`);
        return {
          success: true,
          shifts: response.data,
          count: response.data.length,
          userId: userId,
          message: 'User shifts retrieved successfully'
        };
      } else {
        throw new Error('Failed to get user shifts');
      }
    } catch (error) {
      console.error(`❌ Error getting shifts for user ${userId}:`, error);
      throw new Error(`Failed to get user shifts: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // SHIFT ANALYTICS & UTILITIES
  // ===================

  async getShiftsByDay(dayOfWeek) {
    try {
      const allShifts = await this.getAllUserShifts();
      
      if (allShifts.success) {
        const dayShifts = allShifts.shifts.filter(shift => 
          shift.day_of_week.toLowerCase() === dayOfWeek.toLowerCase()
        );
        
        return {
          success: true,
          shifts: dayShifts,
          count: dayShifts.length,
          day: dayOfWeek,
          message: `Retrieved ${dayShifts.length} shifts for ${dayOfWeek}`
        };
      } else {
        throw new Error('Failed to filter shifts by day');
      }
    } catch (error) {
      console.error(`❌ Error getting shifts for ${dayOfWeek}:`, error);
      throw new Error(`Failed to get shifts for day: ${error.message}`);
    }
  }

  async getShiftStatistics() {
    try {
      const allShifts = await this.getAllUserShifts();
      
      if (allShifts.success) {
        const shifts = allShifts.shifts;
        
        // Calculate statistics
        const totalShifts = shifts.length;
        const dayBreakdown = shifts.reduce((acc, shift) => {
          acc[shift.day_of_week] = (acc[shift.day_of_week] || 0) + 1;
          return acc;
        }, {});
        
        const totalHours = shifts.reduce((total, shift) => {
          const start = new Date(`2000-01-01 ${shift.start_time}`);
          const end = new Date(`2000-01-01 ${shift.end_time}`);
          const hours = (end - start) / (1000 * 60 * 60);
          return total + hours;
        }, 0);
        
        const avgHoursPerShift = totalShifts > 0 ? totalHours / totalShifts : 0;
        
        return {
          success: true,
          statistics: {
            totalShifts,
            totalHours: Math.round(totalHours * 100) / 100,
            avgHoursPerShift: Math.round(avgHoursPerShift * 100) / 100,
            dayBreakdown,
            weeklyHours: Math.round(totalHours * 100) / 100
          },
          message: 'Shift statistics calculated successfully'
        };
      } else {
        throw new Error('Failed to calculate shift statistics');
      }
    } catch (error) {
      console.error('❌ Error calculating shift statistics:', error);
      throw new Error(`Failed to calculate statistics: ${error.message}`);
    }
  }

  // ===================
  // VALIDATION UTILITIES
  // ===================

  validateShiftData(shiftData) {
    const { dayOfWeek, startTime, endTime } = shiftData;
    const errors = [];

    // Validate day of week
    const validDays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'];
    if (!dayOfWeek || !validDays.includes(dayOfWeek.toUpperCase())) {
      errors.push(`Invalid day. Must be one of: ${validDays.join(', ')}`);
    }

    // Validate time format (HH:MM)
    const timeRegex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/;
    if (!startTime || !timeRegex.test(startTime)) {
      errors.push('Start time must be in HH:MM format (e.g., "09:30", "14:00")');
    }
    if (!endTime || !timeRegex.test(endTime)) {
      errors.push('End time must be in HH:MM format (e.g., "09:30", "14:00")');
    }

    // Validate time logic
    if (startTime && endTime && timeRegex.test(startTime) && timeRegex.test(endTime)) {
      const start = new Date(`2000-01-01 ${startTime}`);
      const end = new Date(`2000-01-01 ${endTime}`);
      
      if (start >= end) {
        errors.push('Start time must be before end time');
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  formatShiftTime(shift) {
    if (!shift) return '';
    
    const formatTime = (time) => {
      const [hours, minutes] = time.split(':');
      const hour = parseInt(hours);
      const period = hour >= 12 ? 'PM' : 'AM';
      const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
      return `${displayHour}:${minutes} ${period}`;
    };

    return `${shift.day_name} ${formatTime(shift.start_time)} - ${formatTime(shift.end_time)}`;
  }

  // ===================
  // EVENT LISTENERS
  // ===================

  addShiftListener(callback) {
    const listenerId = Date.now() + Math.random();
    this.eventListeners.set(listenerId, callback);
    return listenerId;
  }

  removeShiftListener(listenerId) {
    this.eventListeners.delete(listenerId);
  }

  notifyShiftListeners(eventType, data) {
    this.eventListeners.forEach(callback => {
      try {
        callback({ type: eventType, data });
      } catch (error) {
        console.error("Error in shift listener callback:", error);
      }
    });
  }

  // ===================
  // CLEANUP
  // ===================

  cleanup = async () => {
    try {
      console.log('🧹 Starting ShiftService cleanup...');
      
      this.eventListeners.clear();
      this.state = ServiceStates.SHUTTING_DOWN;
      
      console.log('✅ ShiftService cleanup completed');
    } catch (error) {
      console.error("❌ Error during ShiftService cleanup:", error);
    }
  };
}