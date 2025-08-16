import api from "../../../../utils/UseAxios";

// Detection Statistics Service - ULTRA-OPTIMIZED VERSION
// Direct API calls with minimal processing for instant response
export class DetectionStatisticsService {
  constructor() {
    this.baseUrl = '/api/detection/basic/statistics';
  }

  // ===================
  // LOT-BASED STATISTICS METHODS - OPTIMIZED
  // ===================

  // Get last detection session for a SPECIFIC lot - INSTANT & DIRECT
  async getLastSessionForLot(lotId) {
    try {
      console.log(`🚀 Getting last session for lot ${lotId} (DIRECT)...`);
      
      const response = await api.get(`${this.baseUrl}/lots/${lotId}/last-session`);

      if (response.data.success) {
        console.log('✅ Last session retrieved instantly');
        
        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error(`Failed to get last session for lot ${lotId}`);
      }
    } catch (error) {
      console.error(`❌ Error getting last session for lot ${lotId}:`, error);
      return {
        success: false,
        data: null,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get last session: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get last detection session for each lot with brief metrics - INSTANT
  async getLastSessionsPerLot() {
    try {
      console.log('📊 Getting last sessions per lot...');
      
      const response = await api.get(`${this.baseUrl}/lots/last-sessions`);

      if (response.data.success) {
        console.log('✅ Last sessions per lot retrieved successfully');
        
        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get last sessions per lot');
      }
    } catch (error) {
      console.error('❌ Error getting last sessions per lot:', error);
      return {
        success: false,
        data: [],
        error: error.response?.data?.detail || error.message,
        message: `Failed to get last sessions per lot: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get summary statistics for a specific lot - INSTANT
  async getLotSummary(lotId) {
    try {
      console.log(`📊 Getting summary for lot ${lotId}...`);

      const response = await api.get(`${this.baseUrl}/lots/${lotId}/summary`);

      if (response.data.success) {
        console.log(`✅ Lot ${lotId} summary retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error(`Failed to get lot ${lotId} summary`);
      }
    } catch (error) {
      console.error(`❌ Error getting lot ${lotId} summary:`, error);
      
      if (error.response?.status === 404) {
        return {
          success: false,
          data: null,
          error: `Lot ${lotId} not found`,
          message: `Lot ${lotId} not found`,
          notFound: true
        };
      }

      return {
        success: false,
        data: null,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get lot summary: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get the number of sessions required to complete a lot - INSTANT
  async getSessionsToCompletion(lotId) {
    try {
      console.log(`📊 Getting sessions to completion for lot ${lotId}...`);

      const response = await api.get(`${this.baseUrl}/lots/${lotId}/sessions-to-completion`);

      if (response.data.success) {
        console.log(`✅ Sessions to completion for lot ${lotId} retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error(`Failed to get sessions to completion for lot ${lotId}`);
      }
    } catch (error) {
      console.error(`❌ Error getting sessions to completion for lot ${lotId}:`, error);
      return {
        success: false,
        data: 0,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get sessions to completion: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // SYSTEM-WIDE STATISTICS METHODS
  // ===================

  // Get stats on lots correctness from their first session - INSTANT
  async getSystemStartStats() {
    try {
      console.log('📊 Getting system start statistics...');

      const response = await api.get(`${this.baseUrl}/system/start-stats`);

      if (response.data.success) {
        console.log('✅ System start statistics retrieved successfully');

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get system start statistics');
      }
    } catch (error) {
      console.error('❌ Error getting system start statistics:', error);
      return {
        success: false,
        data: null,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get system start statistics: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get most common failure categories for problem lots - INSTANT
  async getCommonFailures(topN = 10) {
    try {
      console.log(`📊 Getting top ${topN} common failures...`);

      const response = await api.get(`${this.baseUrl}/system/common-failures?top_n=${topN}`);

      if (response.data.success) {
        console.log(`✅ Top ${topN} common failures retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get common failures');
      }
    } catch (error) {
      console.error('❌ Error getting common failures:', error);
      return {
        success: false,
        data: [],
        error: error.response?.data?.detail || error.message,
        message: `Failed to get common failures: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get most confused piece pairs - INSTANT
  async getTopMixedPairs(topN = 10) {
    try {
      console.log(`📊 Getting top ${topN} mixed-up pairs...`);

      const response = await api.get(`${this.baseUrl}/system/top-mixed-pairs?top_n=${topN}`);

      if (response.data.success) {
        console.log(`✅ Top ${topN} mixed-up pairs retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to get top mixed pairs');
      }
    } catch (error) {
      console.error('❌ Error getting top mixed pairs:', error);
      return {
        success: false,
        data: [],
        error: error.response?.data?.detail || error.message,
        message: `Failed to get top mixed pairs: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // HEALTH CHECK METHODS
  // ===================

  // Check statistics service health - INSTANT
  async checkStatisticsHealth() {
    try {
      console.log('🏥 Checking statistics service health...');

      const response = await api.get(`${this.baseUrl}/health`);

      if (response.data.success) {
        console.log('✅ Statistics service is healthy');

        return {
          success: true,
          data: response.data.data,
          message: response.data.message
        };
      } else {
        throw new Error('Statistics service health check failed');
      }
    } catch (error) {
      console.error('❌ Statistics service health check failed:', error);
      return {
        success: false,
        data: {
          service_status: 'unhealthy'
        },
        error: error.response?.data?.detail || error.message,
        message: `Statistics service unhealthy: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // CONVENIENCE METHODS
  // ===================

  // Get complete dashboard data in one call - INSTANT
  async getDashboardData(options = {}) {
    try {
      console.log('📊 Getting complete dashboard data...');

      const {
        includeLastSessions = true,
        includeSystemStats = true,
        includeCommonFailures = true,
        includeTopMixedPairs = true,
        topN = 10
      } = options;

      const promises = [];

      if (includeLastSessions) {
        promises.push(this.getLastSessionsPerLot());
      }
      if (includeSystemStats) {
        promises.push(this.getSystemStartStats());
      }
      if (includeCommonFailures) {
        promises.push(this.getCommonFailures(topN));
      }
      if (includeTopMixedPairs) {
        promises.push(this.getTopMixedPairs(topN));
      }

      const results = await Promise.allSettled(promises);
      
      let resultIndex = 0;
      const dashboardData = {};
      const errors = {};

      if (includeLastSessions) {
        const lastSessionsResult = results[resultIndex++];
        dashboardData.lastSessions = lastSessionsResult.status === 'fulfilled' ? lastSessionsResult.value.data : null;
        errors.lastSessions = lastSessionsResult.status === 'rejected' ? lastSessionsResult.reason.message : null;
      }

      if (includeSystemStats) {
        const systemStatsResult = results[resultIndex++];
        dashboardData.systemStats = systemStatsResult.status === 'fulfilled' ? systemStatsResult.value.data : null;
        errors.systemStats = systemStatsResult.status === 'rejected' ? systemStatsResult.reason.message : null;
      }

      if (includeCommonFailures) {
        const commonFailuresResult = results[resultIndex++];
        dashboardData.commonFailures = commonFailuresResult.status === 'fulfilled' ? commonFailuresResult.value.data : [];
        errors.commonFailures = commonFailuresResult.status === 'rejected' ? commonFailuresResult.reason.message : null;
      }

      if (includeTopMixedPairs) {
        const topMixedPairsResult = results[resultIndex++];
        dashboardData.topMixedPairs = topMixedPairsResult.status === 'fulfilled' ? topMixedPairsResult.value.data : [];
        errors.topMixedPairs = topMixedPairsResult.status === 'rejected' ? topMixedPairsResult.reason.message : null;
      }

      return {
        success: true,
        data: dashboardData,
        errors: errors,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('❌ Error getting dashboard data:', error);
      return {
        success: false,
        data: null,
        error: error.message,
        message: `Failed to get dashboard data: ${error.message}`
      };
    }
  }

  // Get comprehensive lot analysis - INSTANT
  async getComprehensiveLotAnalysis(lotId) {
    try {
      console.log(`📊 Getting comprehensive analysis for lot ${lotId}...`);

      const promises = [
        this.getLotSummary(lotId),
        this.getSessionsToCompletion(lotId)
      ];

      const [summary, sessionsToCompletion] = await Promise.allSettled(promises);

      return {
        success: true,
        lotId: lotId,
        data: {
          summary: summary.status === 'fulfilled' ? summary.value.data : null,
          sessionsToCompletion: sessionsToCompletion.status === 'fulfilled' ? sessionsToCompletion.value.data : 0
        },
        errors: {
          summary: summary.status === 'rejected' ? summary.reason.message : null,
          sessionsToCompletion: sessionsToCompletion.status === 'rejected' ? sessionsToCompletion.reason.message : null
        },
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error(`❌ Error getting comprehensive analysis for lot ${lotId}:`, error);
      return {
        success: false,
        lotId: lotId,
        data: null,
        error: error.message,
        message: `Failed to get comprehensive lot analysis: ${error.message}`
      };
    }
  }

  // ===================
  // UTILITY METHODS
  // ===================

  // Format statistics data for display
  formatStatsForDisplay(stats) {
    if (!stats) return null;

    return {
      ...stats,
      formattedAccuracy: `${stats.accuracy_percentage || 0}%`,
      formattedCompletion: `${stats.completion_percentage || 0}%`,
      formattedConfidence: `${(stats.confidence_score || 0) * 100}%`,
      formattedDate: stats.created_at ? new Date(stats.created_at).toLocaleDateString() : 'N/A',
      formattedTime: stats.created_at ? new Date(stats.created_at).toLocaleTimeString() : 'N/A'
    };
  }

  // Calculate performance metrics from last sessions data
  calculatePerformanceMetrics(lastSessions) {
    if (!lastSessions || lastSessions.length === 0) {
      return {
        averageAccuracy: 0,
        averageCompletion: 0,
        totalLots: 0,
        completedLots: 0,
        totalSessions: 0,
        totalDetections: 0
      };
    }

    const totals = lastSessions.reduce((acc, session) => ({
      accuracy: acc.accuracy + (session.accuracy_percentage || 0),
      completion: acc.completion + (session.completion_percentage || 0),
      sessions: acc.sessions + 1,
      detections: acc.detections + (session.total_detections || 0),
      completedCount: acc.completedCount + (session.is_completed ? 1 : 0)
    }), { 
      accuracy: 0, 
      completion: 0, 
      sessions: 0, 
      detections: 0, 
      completedCount: 0 
    });

    return {
      averageAccuracy: Math.round((totals.accuracy / lastSessions.length) * 10) / 10,
      averageCompletion: Math.round((totals.completion / lastSessions.length) * 10) / 10,
      totalLots: lastSessions.length,
      completedLots: totals.completedCount,
      totalSessions: totals.sessions,
      totalDetections: totals.detections
    };
  }

  // Format common failures for charts
  formatFailuresForChart(failures) {
    if (!failures || failures.length === 0) {
      return [];
    }

    return failures.map(failure => ({
      name: failure.failure_category || 'Unknown',
      value: failure.count || 0,
      percentage: failure.percentage || 0
    }));
  }

  // Format mixed pairs for display
  formatMixedPairsForDisplay(pairs) {
    if (!pairs || pairs.length === 0) {
      return [];
    }

    return pairs.map(pair => ({
      actualPiece: pair.actual_piece_label || 'Unknown',
      detectedPiece: pair.detected_piece_label || 'Unknown',
      confusionCount: pair.confusion_count || 0,
      percentage: pair.percentage || 0,
      pairDescription: `${pair.actual_piece_label} → ${pair.detected_piece_label}`
    }));
  }

  // Get trend analysis from system stats
  getTrendAnalysis(systemStats) {
    if (!systemStats) {
      return 'unknown';
    }

    const accuracy = systemStats.average_first_session_accuracy || 0;
    
    if (accuracy >= 80) return 'excellent';
    if (accuracy >= 60) return 'good';
    if (accuracy >= 40) return 'fair';
    return 'needs_improvement';
  }

  // Calculate system health score
  calculateSystemHealthScore(dashboardData) {
    if (!dashboardData) return 0;

    let score = 0;
    let factors = 0;

    // Factor 1: System start stats
    if (dashboardData.systemStats) {
      const accuracy = dashboardData.systemStats.average_first_session_accuracy || 0;
      score += Math.min(accuracy, 100);
      factors++;
    }

    // Factor 2: Last sessions performance
    if (dashboardData.lastSessions && dashboardData.lastSessions.length > 0) {
      const metrics = this.calculatePerformanceMetrics(dashboardData.lastSessions);
      score += metrics.averageAccuracy;
      factors++;
    }

    // Factor 3: Failure rate (inverse relationship)
    if (dashboardData.commonFailures && dashboardData.commonFailures.length > 0) {
      const totalFailures = dashboardData.commonFailures.reduce((sum, f) => sum + (f.count || 0), 0);
      const failureScore = Math.max(0, 100 - (totalFailures / 10)); // Assume 10+ failures = 0 score
      score += failureScore;
      factors++;
    }

    return factors > 0 ? Math.round(score / factors) : 0;
  }
}

// Create and export singleton instance
export const detectionStatisticsService = new DetectionStatisticsService();

// Export class for direct instantiation if needed
export default DetectionStatisticsService;