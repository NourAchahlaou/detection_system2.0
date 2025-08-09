import api from "../../../../utils/UseAxios";

// Detection Statistics Service
// Provides comprehensive statistics and analytics for the detection system
// including lot progress, piece statistics, real-time stats, and performance analytics.
export class DetectionStatisticsService {
  constructor() {
    this.baseUrl = '/api/detection/basic/statistics';
    this.cache = new Map();
    this.cacheTimeout = 30000; // 30 seconds cache timeout
  }

  // ===================
  // CACHE MANAGEMENT
  // ===================

  // Check if cached data is still valid
  _isCacheValid(key) {
    const cached = this.cache.get(key);
    if (!cached) return false;
    return (Date.now() - cached.timestamp) < this.cacheTimeout;
  }

  // Get data from cache if valid
  _getFromCache(key) {
    if (this._isCacheValid(key)) {
      return this.cache.get(key).data;
    }
    return null;
  }

  // Store data in cache
  _setCache(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  // Clear all cached statistics data
  clearLocalCache() {
    this.cache.clear();
    console.log('📊 Local statistics cache cleared');
  }

  // ===================
  // OVERVIEW AND SUMMARY METHODS
  // ===================

  // Get overall detection system overview and statistics
  async getDetectionOverview() {
    try {
      const cacheKey = 'detection_overview';
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, data: cached, fromCache: true };
      }

      console.log('📊 Getting detection overview...');
      
      const response = await api.get(`${this.baseUrl}/overview`);

      if (response.data.success) {
        this._setCache(cacheKey, response.data.data);
        console.log('✅ Detection overview retrieved successfully');
        
        return {
          success: true,
          data: response.data.data,
          message: response.data.message,
          fromCache: false
        };
      } else {
        throw new Error('Failed to get detection overview');
      }
    } catch (error) {
      console.error('❌ Error getting detection overview:', error);
      return {
        success: false,
        data: null,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get overview: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get comprehensive detection system summary
  async getDetectionSummary(options = {}) {
    try {
      const {
        includeRealtime = true,
        includeAnalytics = true,
        analyticsDays = 7
      } = options;

      const cacheKey = `detection_summary_${includeRealtime}_${includeAnalytics}_${analyticsDays}`;
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, data: cached, fromCache: true };
      }

      console.log('📊 Getting comprehensive detection summary...');

      const params = new URLSearchParams({
        include_realtime: includeRealtime.toString(),
        include_analytics: includeAnalytics.toString(),
        analytics_days: analyticsDays.toString()
      });

      const response = await api.get(`${this.baseUrl}/summary?${params}`);

      if (response.data.success) {
        this._setCache(cacheKey, response.data.data);
        console.log('✅ Detection summary retrieved successfully');

        return {
          success: true,
          data: response.data.data,
          message: response.data.message,
          fromCache: false
        };
      } else {
        throw new Error('Failed to get detection summary');
      }
    } catch (error) {
      console.error('❌ Error getting detection summary:', error);
      return {
        success: false,
        data: null,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get summary: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // LOT STATISTICS METHODS
  // ===================

  // Get detailed progress information for a specific detection lot
  async getLotProgress(lotId) {
    try {
      const cacheKey = `lot_progress_${lotId}`;
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, data: cached, fromCache: true };
      }

      console.log(`📊 Getting progress for lot ${lotId}...`);

      const response = await api.get(`${this.baseUrl}/lots/${lotId}/progress`);

      if (response.data.success) {
        this._setCache(cacheKey, response.data.data);
        console.log(`✅ Lot ${lotId} progress retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message,
          fromCache: false
        };
      } else {
        throw new Error(`Failed to get lot ${lotId} progress`);
      }
    } catch (error) {
      console.error(`❌ Error getting lot ${lotId} progress:`, error);
      
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
        message: `Failed to get lot progress: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get timeline of detections for a specific lot
  async getLotCompletionTimeline(lotId) {
    try {
      console.log(`📊 Getting completion timeline for lot ${lotId}...`);

      const response = await api.get(`${this.baseUrl}/lots/${lotId}/timeline`);

      if (response.data.success) {
        console.log(`✅ Lot ${lotId} timeline retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          totalSessions: response.data.total_sessions,
          message: response.data.message
        };
      } else {
        throw new Error(`Failed to get lot ${lotId} timeline`);
      }
    } catch (error) {
      console.error(`❌ Error getting lot ${lotId} timeline:`, error);
      return {
        success: false,
        data: [],
        totalSessions: 0,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get lot timeline: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // Get progress information for multiple lots
  async getAllLotsProgress(options = {}) {
    try {
      const {
        limit = 50,
        offset = 0,
        completedOnly = false,
        activeOnly = false
      } = options;

      const cacheKey = `all_lots_progress_${limit}_${offset}_${completedOnly}_${activeOnly}`;
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, ...cached, fromCache: true };
      }

      console.log(`📊 Getting progress for lots (limit: ${limit}, offset: ${offset})...`);

      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
        completed_only: completedOnly.toString(),
        active_only: activeOnly.toString()
      });

      const response = await api.get(`${this.baseUrl}/lots/all/progress?${params}`);

      if (response.data.success) {
        const result = {
          data: response.data.data,
          pagination: response.data.pagination,
          message: response.data.message
        };
        
        this._setCache(cacheKey, result);
        console.log(`✅ Retrieved progress for ${response.data.pagination.returned_count} lots`);

        return {
          success: true,
          ...result,
          fromCache: false
        };
      } else {
        throw new Error('Failed to get lots progress');
      }
    } catch (error) {
      console.error('❌ Error getting all lots progress:', error);
      return {
        success: false,
        data: [],
        pagination: {
          total_count: 0,
          returned_count: 0,
          limit: options.limit || 50,
          offset: options.offset || 0,
          has_more: false
        },
        error: error.response?.data?.detail || error.message,
        message: `Failed to get lots progress: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // PIECE STATISTICS METHODS
  // ===================

  // Get statistics for detected pieces
  async getPieceStatistics(pieceId = null) {
    try {
      const cacheKey = `piece_stats_${pieceId || 'all'}`;
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, ...cached, fromCache: true };
      }

      console.log(`📊 Getting piece statistics${pieceId ? ` for piece ${pieceId}` : ' for all pieces'}...`);

      const params = pieceId ? `?piece_id=${pieceId}` : '';
      const response = await api.get(`${this.baseUrl}/pieces${params}`);

      if (response.data.success) {
        const result = {
          data: response.data.data,
          totalPieces: response.data.total_pieces,
          message: response.data.message
        };

        this._setCache(cacheKey, result);
        console.log('✅ Piece statistics retrieved successfully');

        return {
          success: true,
          ...result,
          fromCache: false
        };
      } else {
        throw new Error('Failed to get piece statistics');
      }
    } catch (error) {
      console.error('❌ Error getting piece statistics:', error);
      return {
        success: false,
        data: [],
        totalPieces: 0,
        error: error.response?.data?.detail || error.message,
        message: `Failed to get piece statistics: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // REAL-TIME STATISTICS METHODS
  // ===================

  // Get real-time detection statistics
  async getRealTimeStats(cameraId = null) {
    try {
      // Don't cache real-time stats as they should be fresh
      console.log(`📊 Getting real-time stats${cameraId ? ` for camera ${cameraId}` : ''}...`);

      const params = cameraId ? `?camera_id=${cameraId}` : '';
      const response = await api.get(`${this.baseUrl}/realtime${params}`);

      if (response.data.success) {
        console.log('✅ Real-time statistics retrieved successfully');

        return {
          success: true,
          data: response.data.data,
          message: response.data.message,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error('Failed to get real-time statistics');
      }
    } catch (error) {
      console.error('❌ Error getting real-time stats:', error);
      return {
        success: false,
        data: {
          current_lot: null,
          recent_detections: [],
          active_cameras: [],
          detection_rate_last_hour: 0.0,
          system_performance: {
            total_sessions_today: 0,
            accuracy_today: 0.0,
            avg_processing_time: 0.0
          }
        },
        error: error.response?.data?.detail || error.message,
        message: `Failed to get real-time statistics: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // PERFORMANCE ANALYTICS METHODS
  // ===================

  // Get performance analytics for the specified number of days
  async getPerformanceAnalytics(days = 7) {
    try {
      const cacheKey = `performance_analytics_${days}`;
      const cached = this._getFromCache(cacheKey);
      if (cached) {
        return { success: true, data: cached, fromCache: true };
      }

      console.log(`📊 Getting performance analytics for last ${days} days...`);

      const response = await api.get(`${this.baseUrl}/analytics?days=${days}`);

      if (response.data.success) {
        this._setCache(cacheKey, response.data.data);
        console.log(`✅ Performance analytics for last ${days} days retrieved successfully`);

        return {
          success: true,
          data: response.data.data,
          message: response.data.message,
          fromCache: false
        };
      } else {
        throw new Error('Failed to get performance analytics');
      }
    } catch (error) {
      console.error('❌ Error getting performance analytics:', error);
      return {
        success: false,
        data: {
          period_days: days,
          daily_performance: [],
          total_sessions: 0,
          total_correct: 0,
          total_misplaced: 0,
          average_confidence: 0.0,
          trend_analysis: {
            accuracy_trend: 'unknown',
            volume_trend: 'unknown',
            confidence_trend: 'unknown'
          }
        },
        error: error.response?.data?.detail || error.message,
        message: `Failed to get performance analytics: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // CACHE MANAGEMENT ENDPOINTS
  // ===================

  // Clear server-side statistics cache
  async clearServerCache() {
    try {
      console.log('🧹 Clearing server statistics cache...');

      const response = await api.delete(`${this.baseUrl}/cache`);

      if (response.data.success) {
        // Also clear local cache
        this.clearLocalCache();
        
        console.log('✅ Server statistics cache cleared successfully');

        return {
          success: true,
          message: response.data.message
        };
      } else {
        throw new Error('Failed to clear server cache');
      }
    } catch (error) {
      console.error('❌ Error clearing server statistics cache:', error);
      return {
        success: false,
        error: error.response?.data?.detail || error.message,
        message: `Failed to clear cache: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // HEALTH CHECK METHODS
  // ===================

  // Check statistics service health
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
          service_status: 'unhealthy',
          cache_entries: 0,
          cache_timeout: 0
        },
        error: error.response?.data?.detail || error.message,
        message: `Statistics service unhealthy: ${error.response?.data?.detail || error.message}`
      };
    }
  }

  // ===================
  // CONVENIENCE METHODS
  // ===================

  // Get complete dashboard data in one call
  async getDashboardData(options = {}) {
    try {
      console.log('📊 Getting complete dashboard data...');

      const promises = [
        this.getDetectionOverview(),
        this.getRealTimeStats(),
        this.getPerformanceAnalytics(options.analyticsDays || 7),
        this.getAllLotsProgress({ limit: 10, activeOnly: true })
      ];

      const [overview, realTime, analytics, activeLots] = await Promise.allSettled(promises);

      return {
        success: true,
        data: {
          overview: overview.status === 'fulfilled' ? overview.value.data : null,
          realTime: realTime.status === 'fulfilled' ? realTime.value.data : null,
          analytics: analytics.status === 'fulfilled' ? analytics.value.data : null,
          activeLots: activeLots.status === 'fulfilled' ? activeLots.value.data : []
        },
        errors: {
          overview: overview.status === 'rejected' ? overview.reason.message : null,
          realTime: realTime.status === 'rejected' ? realTime.reason.message : null,
          analytics: analytics.status === 'rejected' ? analytics.reason.message : null,
          activeLots: activeLots.status === 'rejected' ? activeLots.reason.message : null
        },
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

  // Get lot summary with enhanced details
  async getLotSummary(lotId) {
    try {
      console.log(`📊 Getting complete summary for lot ${lotId}...`);

      const promises = [
        this.getLotProgress(lotId),
        this.getLotCompletionTimeline(lotId)
      ];

      const [progress, timeline] = await Promise.allSettled(promises);

      return {
        success: true,
        lotId: lotId,
        data: {
          progress: progress.status === 'fulfilled' ? progress.value.data : null,
          timeline: timeline.status === 'fulfilled' ? timeline.value.data : []
        },
        errors: {
          progress: progress.status === 'rejected' ? progress.reason.message : null,
          timeline: timeline.status === 'rejected' ? timeline.reason.message : null
        },
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error(`❌ Error getting lot ${lotId} summary:`, error);
      return {
        success: false,
        lotId: lotId,
        data: null,
        error: error.message,
        message: `Failed to get lot summary: ${error.message}`
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
      formattedConfidence: `${(stats.confidence || 0) * 100}%`,
      formattedDate: stats.created_at ? new Date(stats.created_at).toLocaleDateString() : 'N/A',
      formattedTime: stats.last_detection_time ? new Date(stats.last_detection_time).toLocaleTimeString() : 'N/A'
    };
  }

  // Calculate performance metrics
  calculatePerformanceMetrics(dailyPerformance) {
    if (!dailyPerformance || dailyPerformance.length === 0) {
      return {
        averageAccuracy: 0,
        totalSessions: 0,
        totalCorrect: 0,
        totalMisplaced: 0,
        averageConfidence: 0,
        trend: 'stable'
      };
    }

    const totals = dailyPerformance.reduce((acc, day) => ({
      sessions: acc.sessions + day.sessions,
      correct: acc.correct + day.correct_pieces,
      misplaced: acc.misplaced + day.misplaced_pieces,
      confidence: acc.confidence + day.average_confidence
    }), { sessions: 0, correct: 0, misplaced: 0, confidence: 0 });

    const totalPieces = totals.correct + totals.misplaced;
    const averageAccuracy = totalPieces > 0 ? (totals.correct / totalPieces * 100) : 0;
    const averageConfidence = dailyPerformance.length > 0 ? (totals.confidence / dailyPerformance.length) : 0;

    return {
      averageAccuracy: Math.round(averageAccuracy * 10) / 10,
      totalSessions: totals.sessions,
      totalCorrect: totals.correct,
      totalMisplaced: totals.misplaced,
      averageConfidence: Math.round(averageConfidence * 100) / 100,
      trend: this._calculateTrend(dailyPerformance)
    };
  }

  // Calculate trend from daily performance data
  _calculateTrend(dailyPerformance) {
    if (dailyPerformance.length < 2) return 'stable';

    const recent = dailyPerformance.slice(-3);
    const earlier = dailyPerformance.slice(0, Math.max(1, dailyPerformance.length - 3));

    const recentAvg = recent.reduce((acc, day) => acc + day.accuracy_percentage, 0) / recent.length;
    const earlierAvg = earlier.reduce((acc, day) => acc + day.accuracy_percentage, 0) / earlier.length;

    const difference = recentAvg - earlierAvg;
    
    if (difference > 2) return 'improving';
    if (difference < -2) return 'declining';
    return 'stable';
  }
}

// Create and export singleton instance
export const detectionStatisticsService = new DetectionStatisticsService();

// Export class for direct instantiation if needed
export default DetectionStatisticsService;