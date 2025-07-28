// systemPerformanceService.js - Service to determine detection mode
import api from "../../utils/UseAxios";

class SystemPerformanceService {
  constructor() {
    this.performanceMode = null; // 'high' or 'basic'
    this.systemProfile = null;
    this.lastProfileCheck = null;
    this.profileCheckInterval = 30000; // Check every 30 seconds during active use
  }

  /**
   * Get system performance profile from the hardware service
   * This determines whether to use high-performance or basic detection
   */
  async getSystemProfile() {
    try {
      console.log("🔍 Checking system performance profile...");
      
      const response = await api.get('/api/artifact_keeper/system/profile');
      
      this.systemProfile = response.data;
      this.lastProfileCheck = Date.now();
      
      // Determine performance mode based on profile
      this.performanceMode = this.determinePerformanceMode(response.data);
      
      console.log(`✅ System profile detected:`, {
        mode: this.performanceMode,
        profile: this.systemProfile
      });
      
      return {
        mode: this.performanceMode,
        profile: this.systemProfile,
        success: true
      };
      
    } catch (error) {
      console.error("❌ Error getting system profile:", error);
      
      // Default to basic mode if profile check fails
      this.performanceMode = 'basic';
      this.systemProfile = {
        cpu_cores: 2,
        total_memory_gb: 4,
        gpu_available: false,
        performance_score: 30
      };
      
      return {
        mode: this.performanceMode,
        profile: this.systemProfile,
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Determine performance mode based on system specifications
   */
  determinePerformanceMode(profile) {
    const {
      cpu_cores = 2,
      total_memory_gb = 4,
      gpu_available = false,
      performance_score = 30,
      avg_fps = 15
    } = profile;

    // High-performance criteria (adjust thresholds as needed)
    const isHighPerformance = (
      cpu_cores >= 4 &&
      total_memory_gb >= 8 &&
      (gpu_available || performance_score >= 70) &&
      avg_fps >= 20
    );

    return isHighPerformance ? 'high' : 'basic';
  }

  /**
   * Get cached performance mode or fetch new profile
   */
  async getPerformanceMode(forceRefresh = false) {
    const now = Date.now();
    const isStale = !this.lastProfileCheck || 
                   (now - this.lastProfileCheck) > this.profileCheckInterval;

    if (forceRefresh || isStale || !this.performanceMode) {
      await this.getSystemProfile();
    }

    return {
      mode: this.performanceMode,
      profile: this.systemProfile,
      lastCheck: this.lastProfileCheck,
      isStale: isStale
    };
  }

  /**
   * Check if current mode is high performance
   */
  isHighPerformanceMode() {
    return this.performanceMode === 'high';
  }

  /**
   * Check if current mode is basic performance
   */
  isBasicMode() {
    return this.performanceMode === 'basic';
  }

  /**
   * Get performance recommendations
   */
  getPerformanceRecommendations() {
    if (!this.systemProfile) return [];

    const recommendations = [];
    
    if (this.systemProfile.cpu_cores < 4) {
      recommendations.push("Consider upgrading to a CPU with 4+ cores for better performance");
    }
    
    if (this.systemProfile.total_memory_gb < 8) {
      recommendations.push("Adding more RAM (8GB+) will improve detection performance");
    }
    
    if (!this.systemProfile.gpu_available) {
      recommendations.push("A dedicated GPU would significantly improve detection speed");
    }
    
    if (this.systemProfile.avg_fps < 20) {
      recommendations.push("Camera performance is below optimal (20+ FPS recommended)");
    }

    return recommendations;
  }

  /**
   * Get performance mode display info
   */
  getPerformanceModeInfo() {
    const mode = this.performanceMode;
    
    if (mode === 'high') {
      return {
        mode: 'high',
        displayName: 'High Performance',
        description: 'Real-time detection with continuous processing',
        color: 'success',
        icon: '⚡',
        features: [
          'Real-time video streaming',
          'Continuous object detection',
          'Live detection overlays',
          'Automatic target tracking'
        ]
      };
    } else {
      return {
        mode: 'basic',
        displayName: 'Basic Mode',
        description: 'On-demand detection with manual triggers',
        color: 'info',
        icon: '🎯',
        features: [
          'Real-time video streaming',
          'Manual detection triggers',
          'Single-frame analysis',
          'Optimized for lower-end hardware'
        ]
      };
    }
  }

  /**
   * Reset performance mode (for manual override or testing)
   */
  resetPerformanceMode() {
    this.performanceMode = null;
    this.systemProfile = null;
    this.lastProfileCheck = null;
    console.log("🔄 Performance mode reset");
  }

  /**
   * Manually set performance mode (for testing or user override)
   */
  setPerformanceMode(mode) {
    if (mode === 'high' || mode === 'basic') {
      this.performanceMode = mode;
      console.log(`🔧 Performance mode manually set to: ${mode}`);
      return true;
    }
    return false;
  }

  /**
   * Get system health for the current mode
   */
  async checkSystemHealth() {
    try {
      if (this.performanceMode === 'high') {
        // Check high-performance services health
        const response = await api.get('/api/detection/redis/health');
        return {
          mode: 'high',
          healthy: response.data.status === 'healthy',
          details: response.data
        };
      } else {
        // Check basic detection service health
        const response = await api.get('/api/detection/basic/health');
        return {
          mode: 'basic',
          healthy: response.data.status === 'healthy',
          details: response.data
        };
      }
    } catch (error) {
      return {
        mode: this.performanceMode,
        healthy: false,
        error: error.message
      };
    }
  }
}

// Export singleton instance
export const systemPerformanceService = new SystemPerformanceService();