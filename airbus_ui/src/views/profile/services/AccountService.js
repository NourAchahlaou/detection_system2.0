import api from "../../../utils/UseAxios";

// Define the 4 states clearly
const ServiceStates = {
  INITIALIZING: 'INITIALIZING',
  READY: 'READY',
  RUNNING: 'RUNNING',
  SHUTTING_DOWN: 'SHUTTING_DOWN'
};

export class ProfileService {
  constructor() {
    this.state = ServiceStates.READY;
    this.eventListeners = new Map();
    this.currentProfile = null;
  }

  // ===================
  // PROFILE OPERATIONS
  // ===================

  async getUserProfile() {
    try {
      console.log('👤 Getting user profile...');
      
      const response = await api.get('/api/users/profile/');
      
      if (response.data) {
        this.currentProfile = response.data;
        this.notifyProfileListeners('profile_updated', response.data);
        
        console.log(`✅ Retrieved profile for ${response.data.name} (${response.data.email})`);
        return {
          success: true,
          profile: response.data,
          message: 'Profile retrieved successfully'
        };
      } else {
        throw new Error('Failed to get user profile');
      }
    } catch (error) {
      console.error('❌ Error getting user profile:', error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message;
      
      return {
        success: false,
        profile: null,
        message: errorMessage,
        error: errorMessage
      };
    }
  }

  async getBasicUserInfo() {
    try {
      console.log('👤 Getting basic user info...');
      
      const response = await api.get('/api/users/profile/basic');
      
      if (response.data) {
        console.log(`✅ Retrieved basic info for ${response.data.role}: ${response.data.name}`);
        console.log('Basic Info:', response.data);
        return {
          success: true,
          profile: response.data,
          message: 'Basic profile retrieved successfully'
        };
      } else {
        throw new Error('Failed to get basic user info');
      }
    } catch (error) {
      console.error('❌ Error getting basic user info:', error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message;
      
      return {
        success: false,
        profile: null,
        message: errorMessage,
        error: errorMessage
      };
    }
  }

  async updateUserProfile(updateData) {
    try {
      console.log('👤 Updating user profile...');
      
      // Build request body with only provided fields
      const requestBody = {};
      if (updateData.name !== undefined) requestBody.name = updateData.name;
      if (updateData.email !== undefined) requestBody.email = updateData.email;
      if (updateData.password !== undefined) requestBody.password = updateData.password;
      if (updateData.airbusId !== undefined) requestBody.airbus_id = updateData.airbusId;
      if (updateData.role !== undefined) requestBody.role = updateData.role;
      
      const response = await api.put('/api/users/profile/', requestBody);
      
      if (response.data) {
        this.currentProfile = response.data;
        this.notifyProfileListeners('profile_updated', response.data);
        
        console.log(`✅ Updated profile for ${response.data.name}`);
        return {
          success: true,
          profile: response.data,
          message: 'Profile updated successfully'
        };
      } else {
        throw new Error('Failed to update profile');
      }
    } catch (error) {
      console.error('❌ Error updating profile:', error);
      throw new Error(`Failed to update profile: ${error.response?.data?.detail || error.message}`);
    }
  }

  async deleteUserAccount() {
    try {
      console.log('👤 Deleting user account...');
      
      const response = await api.delete('/api/users/profile/');
      
      if (response.data) {
        this.currentProfile = null;
        this.notifyProfileListeners('account_deleted', null);
        
        console.log('✅ User account deleted successfully');
        return {
          success: true,
          message: response.data.message || 'Account deleted successfully'
        };
      } else {
        throw new Error('Failed to delete account');
      }
    } catch (error) {
      console.error('❌ Error deleting account:', error);
      throw new Error(`Failed to delete account: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // ADMIN OPERATIONS
  // ===================

  async getUserBasicInfoById(userId) {
    try {
      console.log(`👤 Getting basic info for user ${userId}...`);
      
      const response = await api.get(`/api/users/profile/${userId}`);
      
      if (response.data) {
        console.log(`✅ Retrieved basic info for user ${userId}: ${response.data.name}`);
        return {
          success: true,
          profile: response.data,
          userId: userId,
          message: 'User profile retrieved successfully'
        };
      } else {
        throw new Error('Failed to get user profile');
      }
    } catch (error) {
      console.error(`❌ Error getting profile for user ${userId}:`, error);
      throw new Error(`Failed to get user profile: ${error.response?.data?.detail || error.message}`);
    }
  }

  // ===================
  // PROFILE ANALYTICS & UTILITIES
  // ===================

  async getProfileStats() {
    try {
      const profileResult = await this.getUserProfile();
      
      if (profileResult.success) {
        const profile = profileResult.profile;
        
        // Calculate profile completeness
        const fields = ['name', 'email', 'airbus_id', 'role'];
        const completedFields = fields.filter(field => 
          profile[field] && profile[field] !== null && profile[field] !== ''
        );
        
        const completeness = Math.round((completedFields.length / fields.length) * 100);
        
        // Calculate account age in days
        let accountAgeDays = 0;
        if (profile.created_at) {
          const createdDate = new Date(profile.created_at);
          const now = new Date();
          accountAgeDays = Math.floor((now - createdDate) / (1000 * 60 * 60 * 24));
        }
        
        return {
          success: true,
          stats: {
            completeness,
            completedFields: completedFields.length,
            totalFields: fields.length,
            shiftsCount: profile.shifts_count || 0,
            accountAgeDays,
            isActive: profile.is_active,
            isVerified: !!profile.verified_at,
            lastUpdated: profile.updated_at,
            memberSince: profile.created_at
          },
          message: 'Profile statistics calculated successfully'
        };
      } else {
        throw new Error('Failed to calculate profile statistics');
      }
    } catch (error) {
      console.error('❌ Error calculating profile stats:', error);
      throw new Error(`Failed to calculate statistics: ${error.message}`);
    }
  }

  // ===================
  // VALIDATION UTILITIES
  // ===================

  validateProfileData(profileData) {
    const errors = [];

    // Validate name
    if (profileData.name !== undefined) {
      if (typeof profileData.name !== 'string' || profileData.name.trim().length === 0) {
        errors.push('Name cannot be empty');
      }
      if (profileData.name.trim().length > 100) {
        errors.push('Name must be less than 100 characters');
      }
    }

    // Validate email
    if (profileData.email !== undefined) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(profileData.email)) {
        errors.push('Invalid email format');
      }
      if (profileData.email.length > 255) {
        errors.push('Email must be less than 255 characters');
      }
    }

    // Validate password
    if (profileData.password !== undefined) {
      if (profileData.password.length < 6) {
        errors.push('Password must be at least 6 characters long');
      }
      if (profileData.password.length > 128) {
        errors.push('Password must be less than 128 characters');
      }
    }

    // Validate airbus_id
    if (profileData.airbusId !== undefined) {
      if (!Number.isInteger(profileData.airbusId) || profileData.airbusId <= 0) {
        errors.push('Airbus ID must be a positive number');
      }
    }

    // Validate role
    if (profileData.role !== undefined) {
      const validRoles = ['USER', 'ADMIN'];
      if (!validRoles.includes(profileData.role.toUpperCase())) {
        errors.push(`Invalid role. Valid roles are: ${validRoles.join(', ')}`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  checkPasswordStrength(password) {
    if (!password) return { strength: 'none', score: 0, suggestions: ['Password is required'] };

    const checks = {
      length: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      numbers: /\d/.test(password),
      symbols: /[^A-Za-z0-9]/.test(password)
    };

    const score = Object.values(checks).reduce((acc, check) => acc + (check ? 1 : 0), 0);
    
    let strength = 'weak';
    if (score >= 4) strength = 'strong';
    else if (score >= 3) strength = 'medium';

    const suggestions = [];
    if (!checks.length) suggestions.push('Use at least 8 characters');
    if (!checks.uppercase) suggestions.push('Add uppercase letters');
    if (!checks.lowercase) suggestions.push('Add lowercase letters');
    if (!checks.numbers) suggestions.push('Add numbers');
    if (!checks.symbols) suggestions.push('Add special characters');

    return { strength, score, suggestions };
  }

  // ===================
  // PROFILE FORMATTING
  // ===================

  formatProfileForDisplay(profile) {
    if (!profile) return null;

    return {
      ...profile,
      displayName: profile.name || 'Unknown User',
      displayRole: profile.role ? profile.role.charAt(0) + profile.role.slice(1).toLowerCase() : 'User',
      accountAge: this.calculateAccountAge(profile.created_at),
      lastActiveFormatted: this.formatDate(profile.updated_at),
      joinedFormatted: this.formatDate(profile.created_at),
      verificationStatus: profile.verified_at ? 'Verified' : 'Not Verified'
    };
  }

  calculateAccountAge(createdAt) {
    if (!createdAt) return 'Unknown';
    
    const created = new Date(createdAt);
    const now = new Date();
    const diffTime = Math.abs(now - created);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return '1 day';
    if (diffDays < 30) return `${diffDays} days`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months`;
    return `${Math.floor(diffDays / 365)} years`;
  }

  formatDate(dateString) {
    if (!dateString) return 'Never';
    
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  // ===================
  // PROFILE CACHING
  // ===================

  getCachedProfile() {
    return this.currentProfile;
  }

  async refreshProfile() {
    console.log('🔄 Refreshing profile cache...');
    return await this.getUserProfile();
  }

  clearProfileCache() {
    this.currentProfile = null;
    this.notifyProfileListeners('cache_cleared', null);
  }

  // ===================
  // EVENT LISTENERS
  // ===================

  addProfileListener(callback) {
    const listenerId = Date.now() + Math.random();
    this.eventListeners.set(listenerId, callback);
    return listenerId;
  }

  removeProfileListener(listenerId) {
    this.eventListeners.delete(listenerId);
  }

  notifyProfileListeners(eventType, data) {
    this.eventListeners.forEach(callback => {
      try {
        callback({ type: eventType, data, timestamp: Date.now() });
      } catch (error) {
        console.error("Error in profile listener callback:", error);
      }
    });
  }

  // ===================
  // BULK OPERATIONS
  // ===================

  async updateMultipleFields(updates) {
    try {
      console.log(`👤 Updating ${Object.keys(updates).length} profile fields...`);
      
      // Validate all fields first
      const validation = this.validateProfileData(updates);
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
      }
      
      return await this.updateUserProfile(updates);
    } catch (error) {
      console.error('❌ Error updating multiple fields:', error);
      throw error;
    }
  }

  async changePassword(currentPassword, newPassword) {
    try {
      console.log('🔐 Changing password...');
      
      // Validate new password
      const passwordCheck = this.checkPasswordStrength(newPassword);
      if (passwordCheck.strength === 'weak') {
        throw new Error(`Password is too weak. ${passwordCheck.suggestions.join(', ')}`);
      }
      
      return await this.updateUserProfile({ password: newPassword });
    } catch (error) {
      console.error('❌ Error changing password:', error);
      throw error;
    }
  }

  // ===================
  // CLEANUP
  // ===================

  cleanup = async () => {
    try {
      console.log('🧹 Starting ProfileService cleanup...');
      
      this.eventListeners.clear();
      this.currentProfile = null;
      this.state = ServiceStates.SHUTTING_DOWN;
      
      console.log('✅ ProfileService cleanup completed');
    } catch (error) {
      console.error("❌ Error during ProfileService cleanup:", error);
    }
  };
}