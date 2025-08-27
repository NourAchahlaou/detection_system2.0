// lotSessionService.js - Frontend service for lot session dashboard
class LotSessionService {
  constructor() {
    this.baseURL = '/api/detection/lotSession';
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  async fetchWithErrorHandling(url, options = {}) {
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  getCacheKey(endpoint, params = {}) {
    const sortedParams = Object.keys(params)
      .sort()
      .map(key => `${key}=${params[key]}`)
      .join('&');
    return `${endpoint}?${sortedParams}`;
  }

  isValidCacheEntry(entry) {
    return entry && (Date.now() - entry.timestamp) < this.cacheTimeout;
  }

  async getDashboardData(filters = {}) {
    const params = new URLSearchParams();
    
    if (filters.groupFilter) params.append('group_filter', filters.groupFilter);
    if (filters.search) params.append('search', filters.search);
    if (filters.statusFilter) params.append('status_filter', filters.statusFilter);

    const cacheKey = this.getCacheKey('data', Object.fromEntries(params));
    const cachedEntry = this.cache.get(cacheKey);

    if (this.isValidCacheEntry(cachedEntry)) {
      return cachedEntry.data;
    }

    const url = `${this.baseURL}/data${params.toString() ? `?${params.toString()}` : ''}`;
    const data = await this.fetchWithErrorHandling(url);

    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });

    return data;
  }

  async getLotDetails(lotId) {
    if (!lotId) {
      throw new Error('Lot ID is required');
    }

    const cacheKey = this.getCacheKey(`lots/${lotId}/details`);
    const cachedEntry = this.cache.get(cacheKey);

    if (this.isValidCacheEntry(cachedEntry)) {
      return cachedEntry.data;
    }

    const url = `${this.baseURL}/lots/${lotId}/details`;
    const data = await this.fetchWithErrorHandling(url);

    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });

    return data;
  }

  async getGroupSummary(groupName) {
    if (!groupName) {
      throw new Error('Group name is required');
    }

    const cacheKey = this.getCacheKey(`groups/${groupName}/summary`);
    const cachedEntry = this.cache.get(cacheKey);

    if (this.isValidCacheEntry(cachedEntry)) {
      return cachedEntry.data;
    }

    const url = `${this.baseURL}/groups/${encodeURIComponent(groupName)}/summary`;
    const data = await this.fetchWithErrorHandling(url);

    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });

    return data;
  }

  async getGroups() {
    const cacheKey = this.getCacheKey('groups');
    const cachedEntry = this.cache.get(cacheKey);

    if (this.isValidCacheEntry(cachedEntry)) {
      return cachedEntry.data;
    }

    const url = `${this.baseURL}/groups`;
    const data = await this.fetchWithErrorHandling(url);

    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });

    return data;
  }

  async getStatisticsOverview() {
    const cacheKey = this.getCacheKey('statistics/overview');
    const cachedEntry = this.cache.get(cacheKey);

    if (this.isValidCacheEntry(cachedEntry)) {
      return cachedEntry.data;
    }

    const url = `${this.baseURL}/statistics/overview`;
    const data = await this.fetchWithErrorHandling(url);

    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });

    return data;
  }

  async checkHealth() {
    const url = `${this.baseURL}/health`;
    return await this.fetchWithErrorHandling(url);
  }

  clearCache() {
    this.cache.clear();
  }

  clearCacheEntry(endpoint, params = {}) {
    const cacheKey = this.getCacheKey(endpoint, params);
    this.cache.delete(cacheKey);
  }

  // Helper methods for data processing
  processGroupedData(dashboardData) {
    if (!dashboardData?.success || !dashboardData?.groupedLots) {
      return {};
    }

    const groupedLots = {};
    
    Object.entries(dashboardData.groupedLots).forEach(([groupName, lots]) => {
      const groupStats = dashboardData.groupStats?.[groupName] || {};
      
      groupedLots[groupName] = {
        groupName,
        lots,
        totalLots: lots.length,
        totalSessions: lots.reduce((acc, lot) => acc + (lot.totalSessions || 0), 0),
        avgSuccessRate: groupStats.avgSessionSuccessRate || 0,
        avgConfidence: groupStats.avgLotConfidence || 0,
        lastActivity: groupStats.lastActivity ? new Date(groupStats.lastActivity) : new Date()
      };
    });

    return groupedLots;
  }

  processStatistics(dashboardData) {
    if (!dashboardData?.success || !dashboardData?.statistics) {
      return {
        totalGroups: 0,
        totalLots: 0,
        totalSessions: 0,
        avgSuccessRate: 0,
        avgConfidence: 0,
        activeGroups: 0
      };
    }

    return {
      totalGroups: dashboardData.statistics.totalGroups || 0,
      totalLots: dashboardData.statistics.totalLots || 0,
      totalSessions: dashboardData.statistics.totalSessions || 0,
      avgSuccessRate: dashboardData.statistics.sessionSuccessRate || 0,
      avgConfidence: dashboardData.statistics.avgLotConfidence || 0,
      activeGroups: dashboardData.statistics.activeGroups || 0
    };
  }

  // Format helpers
  formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return '';
    
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  formatPercentage(value, decimals = 1) {
    if (typeof value !== 'number' || isNaN(value)) return '0.0%';
    return `${value.toFixed(decimals)}%`;
  }
}

// Create a singleton instance
const lotSessionService = new LotSessionService();

export default lotSessionService;