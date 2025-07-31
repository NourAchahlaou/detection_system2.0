import api from "../../../utils/UseAxios";

// Performance modes based on system capabilities
const PerformanceModes = {
  BASIC: 'basic',
  STANDARD: 'standard', 
  ENHANCED: 'enhanced',
  HIGH_PERFORMANCE: 'high_performance'
};

// Streaming types
const StreamingTypes = {
  BASIC: 'basic',
  OPTIMIZED: 'optimized'
};

export class SystemProfiler {
  constructor(detectionService) {
    this.detectionService = detectionService;
  }

  async initializeSystemProfiling() {
    try {
      console.log('🖥️ Initializing system profiling...');
      await this.updateSystemProfile();
      console.log(`✅ System profiling initialized - Mode: ${this.detectionService.currentPerformanceMode}, Streaming: ${this.detectionService.currentStreamingType}`);
    } catch (error) {
      console.error('⚠️ Failed to initialize system profiling:', error);
      // Continue with basic mode as fallback
      this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
      this.detectionService.currentStreamingType = StreamingTypes.BASIC;
    }
  }

  async updateSystemProfile(forceRefresh = false) {
    try {
      // Check cache first
      if (!forceRefresh && this.detectionService.systemProfile && 
          this.detectionService.systemProfile.timestamp && 
          (Date.now() - new Date(this.detectionService.systemProfile.timestamp).getTime()) < this.detectionService.SYSTEM_PROFILE_CACHE_DURATION) {
        console.log('📊 Using cached system profile');
        return this.detectionService.systemProfile;
      }

      console.log('🔍 Fetching system profile from artifact keeper...');
      
      // Get system profile from artifact keeper
      const profileResponse = await api.get('/api/artifact_keeper/system/profile', {
        params: { force_refresh: forceRefresh }
      });
      
      this.detectionService.systemProfile = profileResponse.data;
      
      // Get performance recommendation
      const recommendationResponse = await api.get('/api/artifact_keeper/system/performance/recommendation');
      const recommendation = recommendationResponse.data;
      
      // Get system capabilities
      const capabilitiesResponse = await api.get('/api/artifact_keeper/system/capabilities');
      this.detectionService.systemCapabilities = capabilitiesResponse.data;
      
      // Determine optimal performance mode and streaming type
      this.determineOptimalSettings(this.detectionService.systemProfile, recommendation, this.detectionService.systemCapabilities);
      
      // Notify listeners
      this.detectionService.notifyProfileUpdateListeners();
      
      console.log(`📊 System profile updated - Performance: ${this.detectionService.currentPerformanceMode}, Streaming: ${this.detectionService.currentStreamingType}`);
      console.log(`   CPU: ${this.detectionService.systemProfile.cpu_cores} cores @ ${this.detectionService.systemProfile.cpu_frequency_mhz}MHz`);
      console.log(`   Memory: ${this.detectionService.systemProfile.available_memory_gb}GB available`);
      console.log(`   GPU: ${this.detectionService.systemProfile.gpu_available ? this.detectionService.systemProfile.gpu_name : 'None'}`);
      console.log(`   Score: ${this.detectionService.systemProfile.performance_score}/100`);
      
      return this.detectionService.systemProfile;
      
    } catch (error) {
      console.error('❌ Error updating system profile:', error);
      
      // Fallback to basic mode if profiling fails
      if (!this.detectionService.systemProfile) {
        this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
        this.detectionService.currentStreamingType = StreamingTypes.BASIC;
        console.log('⚠️ Using fallback basic mode due to profiling error');
      }
      
      throw error;
    }
  }

  determineOptimalSettings(profile, recommendation, capabilities) {
    try {
      const score = profile.performance_score || 0;
      const recommendedMode = recommendation.final_recommendation || 'basic';
      const meetsMinimum = profile.meets_minimum_requirements || false;
      const gpuAvailable = profile.gpu_available || false;
      const cudaAvailable = profile.cuda_available || false;
      
      // Determine performance mode based on system specs
      if (score >= 80 && gpuAvailable && cudaAvailable && meetsMinimum) {
        this.detectionService.currentPerformanceMode = PerformanceModes.HIGH_PERFORMANCE;
        this.detectionService.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else if (score >= 60 && gpuAvailable && meetsMinimum) {
        this.detectionService.currentPerformanceMode = PerformanceModes.ENHANCED;
        this.detectionService.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else if (score >= 40 && meetsMinimum) {
        this.detectionService.currentPerformanceMode = PerformanceModes.STANDARD;
        this.detectionService.currentStreamingType = StreamingTypes.OPTIMIZED;
      } else {
        this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
        this.detectionService.currentStreamingType = StreamingTypes.BASIC;
      }
      
      // Override with explicit recommendation if available
      if (recommendedMode === 'basic') {
        this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
        this.detectionService.currentStreamingType = StreamingTypes.BASIC;
      } else if (recommendedMode === 'standard' && this.detectionService.currentPerformanceMode === PerformanceModes.BASIC) {
        this.detectionService.currentPerformanceMode = PerformanceModes.STANDARD;
        this.detectionService.currentStreamingType = StreamingTypes.OPTIMIZED;
      }
      
      console.log(`🎯 Determined optimal settings based on score ${score}:`);
      console.log(`   Performance Mode: ${this.detectionService.currentPerformanceMode}`);
      console.log(`   Streaming Type: ${this.detectionService.currentStreamingType}`);
      console.log(`   Recommended Mode: ${recommendedMode}`);
      
    } catch (error) {
      console.error('❌ Error determining optimal settings:', error);
      // Fallback to basic
      this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
      this.detectionService.currentStreamingType = StreamingTypes.BASIC;
    }
  }

  async forceSystemProfileRefresh() {
    try {
      console.log('🔄 Force refreshing system profile...');
      
      // Force refresh cache on backend
      await api.post('/api/artifact_keeper/system/cache/refresh');
      
      // Update local profile
      await this.updateSystemProfile(true);
      
      return {
        success: true,
        profile: this.detectionService.systemProfile,
        performanceMode: this.detectionService.currentPerformanceMode,
        streamingType: this.detectionService.currentStreamingType
      };
    } catch (error) {
      console.error('❌ Error force refreshing system profile:', error);
      throw error;
    }
  }

  async runPerformanceTest(durationSeconds = 10) {
    try {
      console.log(`🧪 Running performance test for ${durationSeconds} seconds...`);
      
      const response = await api.post('/api/artifact_keeper/system/performance/test', null, {
        params: { duration_seconds: durationSeconds }
      });
      
      // Update profile after test
      await this.updateSystemProfile(true);
      
      return response.data;
    } catch (error) {
      console.error('❌ Error running performance test:', error);
      throw error;
    }
  }

  async monitorCameraPerformance(cameraId, durationSeconds = 5) {
    try {
      console.log(`📊 Monitoring camera ${cameraId} performance for ${durationSeconds} seconds...`);
      
      const response = await api.get(`/api/artifact_keeper/system/performance/monitor/${cameraId}`, {
        params: { duration_seconds: durationSeconds }
      });
      
      return response.data;
    } catch (error) {
      console.error(`❌ Error monitoring camera ${cameraId} performance:`, error);
      throw error;
    }
  }

  // Manual mode switching
  async switchToBasicMode() {
    try {
      console.log('🔄 Manually switching to basic mode...');
      
      // Stop all current streams and unfreeze any frozen ones
      await this.detectionService.stopAllStreams(false);
      
      this.detectionService.currentPerformanceMode = PerformanceModes.BASIC;
      this.detectionService.currentStreamingType = StreamingTypes.BASIC;
      this.detectionService.autoModeEnabled = false;
      
      this.detectionService.notifyProfileUpdateListeners();
      
      console.log('✅ Switched to basic mode');
      return { success: true, mode: 'basic' };
    } catch (error) {
      console.error('❌ Error switching to basic mode:', error);
      throw error;
    }
  }

  async switchToOptimizedMode() {
    try {
      console.log('🔄 Manually switching to optimized mode...');
      
      // Check if system supports optimized mode
      if (!this.detectionService.systemProfile || this.detectionService.systemProfile.performance_score < 30) {
        throw new Error('System does not meet minimum requirements for optimized mode');
      }
      
      // Stop all current streams and unfreeze any frozen ones
      await this.detectionService.stopAllStreams(false);
      
      this.detectionService.currentStreamingType = StreamingTypes.OPTIMIZED;
      if (this.detectionService.currentPerformanceMode === PerformanceModes.BASIC) {
        this.detectionService.currentPerformanceMode = PerformanceModes.STANDARD;
      }
      this.detectionService.autoModeEnabled = false;
      
      this.detectionService.notifyProfileUpdateListeners();
      
      console.log('✅ Switched to optimized mode');
      return { success: true, mode: 'optimized' };
    } catch (error) {
      console.error('❌ Error switching to optimized mode:', error);
      throw error;
    }
  }

  async enableAutoMode() {
    try {
      console.log('🤖 Enabling automatic mode selection...');
      this.detectionService.autoModeEnabled = true;
      
      // Re-evaluate optimal settings
      await this.updateSystemProfile(true);
      
      console.log(`✅ Auto mode enabled - Selected: ${this.detectionService.currentStreamingType}`);
      return { success: true, mode: this.detectionService.currentStreamingType };
    } catch (error) {
      console.error('❌ Error enabling auto mode:', error);
      throw error;
    }
  }
}