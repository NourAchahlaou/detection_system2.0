import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Avatar,
  Grid,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Collapse,
  InputAdornment,
  TablePagination,
  Tooltip,
  CircularProgress,
  Alert,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import {
  ExpandMore,
  Group as GroupIcon,
  Search,
  FilterList,
  Clear,
  Refresh,
  CheckCircle,
  Error,
  Schedule,
  Analytics,
  Camera,
  Visibility,
  DateRange
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import lotSessionService from './lotSessionService';

// Styled Components (keeping all your existing styles)
const Container = styled(Box)(({ theme }) => ({
  padding: '24px',
  backgroundColor: 'transparent',
  minHeight: '100vh',
}));

const HeaderBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '24px',
  padding: '16px 24px',
  backgroundColor: 'transparent',
  borderRadius: '12px',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}));

const StatsCard = styled(Card)(({ theme }) => ({
  borderRadius: '12px',
  boxShadow: '0 2px 8px rgba(102, 126, 234, 0.1)',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  '&:hover': {
    transform: 'translateY(-2px)',
    transition: 'all 0.2s ease-in-out',
    boxShadow: '0 4px 16px rgba(102, 126, 234, 0.15)',
  }
}));

const FilterCard = styled(Card)(({ theme }) => ({
  marginBottom: '24px',
  borderRadius: '12px',
  border: '1px solid rgba(102, 126, 234, 0.1)',
}));

const GroupAccordion = styled(Accordion)(({ theme }) => ({
  marginBottom: '16px',
  borderRadius: '12px !important',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  '&:before': {
    display: 'none',
  },
  boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
}));

const SessionTable = styled(Table)(({ theme }) => ({
  '& .MuiTableHead-root': {
    backgroundColor: 'rgba(102, 126, 234, 0.02)',
  },
  '& .MuiTableCell-head': {
    fontWeight: 600,
    color: '#667eea',
    borderBottom: '2px solid rgba(102, 126, 234, 0.1)',
  },
  '& .MuiTableRow-root:hover': {
    backgroundColor: 'rgba(102, 126, 234, 0.02)',
  },
}));

const StatusChip = styled(Chip)(({ variant }) => ({
  fontWeight: 600,
  fontSize: '0.75rem',
  ...(variant === 'completed' && {
    backgroundColor: '#e8f5e8',
    color: '#2e7d2e',
  }),
  ...(variant === 'failed' && {
    backgroundColor: '#ffebee',
    color: '#c62828',
  }),
  ...(variant === 'running' && {
    backgroundColor: '#fff3e0',
    color: '#ef6c00',
  }),
  ...(variant === 'pending' && {
    backgroundColor: '#f3f4f6',
    color: '#6b7280',
  }),
}));

// Mock data generator (keeping your existing logic as fallback)
const generateMockData = () => {
  const lotGroups = ['PROD_A', 'PROD_B', 'TEST_X', 'QUAL_Y'];
  const cameras = [1, 2, 3, 4];
  const statuses = ['completed', 'failed', 'running', 'pending'];
  const pieces = ['piece_001', 'piece_002', 'piece_003', 'piece_004', 'piece_005'];
  
  const lots = [];
  
  lotGroups.forEach(group => {
    const lotCount = Math.floor(Math.random() * 5) + 3; // 3-7 lots per group
    
    for (let i = 0; i < lotCount; i++) {
      const lotId = `${group}_LOT_${String(i + 1).padStart(3, '0')}`;
      const sessionCount = Math.floor(Math.random() * 8) + 2; // 2-9 sessions per lot
      
      const sessions = [];
      for (let j = 0; j < sessionCount; j++) {
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        const camera = cameras[Math.floor(Math.random() * cameras.length)];
        const piece = pieces[Math.floor(Math.random() * pieces.length)];
        const confidence = status === 'completed' ? Math.random() * 0.3 + 0.7 : Math.random() * 0.8;
        const isMatch = status === 'completed' ? Math.random() > 0.3 : false;
        
        sessions.push({
          id: `session_${lotId}_${j + 1}`,
          sessionNumber: j + 1,
          cameraId: camera,
          targetPiece: piece,
          detectedPiece: isMatch ? piece : pieces[Math.floor(Math.random() * pieces.length)],
          confidence: confidence * 100, // Convert to percentage
          isTargetMatch: isMatch,
          status: status,
          timestamp: new Date(Date.now() - Math.random() * 86400000 * 7), // Last 7 days
          processingTime: Math.floor(Math.random() * 2000) + 500, // 500-2500ms
          detectionRate: Math.random() * 0.4 + 0.6 // 0.6-1.0
        });
      }
      
      const completedSessions = sessions.filter(s => s.status === 'completed').length;
      const matchingSessions = sessions.filter(s => s.isTargetMatch && s.status === 'completed').length;
      
      // Fix: Calculate average confidence properly
      const completedSessionsWithConfidence = sessions.filter(s => s.status === 'completed');
      const avgConfidence = completedSessionsWithConfidence.length > 0 
        ? completedSessionsWithConfidence.reduce((acc, s) => acc + s.confidence, 0) / completedSessionsWithConfidence.length
        : 0;
      
      lots.push({
        id: lotId,
        group: group,
        lotName: lotId.replace(group + '_', ''),
        expectedPiece: pieces[Math.floor(Math.random() * pieces.length)],
        expectedPieceNumber: Math.floor(Math.random() * 100) + 1,
        sessions: sessions,
        totalSessions: sessions.length,
        completedSessions: completedSessions,
        successfulMatches: matchingSessions,
        successRate: completedSessions > 0 ? (matchingSessions / completedSessions * 100).toFixed(1) : 0,
        avgConfidence: avgConfidence.toFixed(1),
        createdAt: new Date(Date.now() - Math.random() * 86400000 * 30), // Last 30 days
        lastActivity: sessions.length > 0 ? new Date(Math.max(...sessions.map(s => s.timestamp.getTime()))) : new Date(),
        // Add lot-level status based on sessions
        lotStatus: sessions.some(s => s.status === 'running') ? 'running' : 
                  sessions.some(s => s.status === 'pending') ? 'pending' :
                  sessions.every(s => s.status === 'completed') ? 'completed' : 'mixed'
      });
    }
  });
  
  return lots;
};

export default function LotSessionDatabase() {
  // Enhanced state management
  const [lots, setLots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [useMockData, setUseMockData] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [expandedLots, setExpandedLots] = useState(new Set());
  const [dashboardData, setDashboardData] = useState(null);
  const [statistics, setStatistics] = useState({
    totalGroups: 0,
    totalLots: 0,
    totalSessions: 0,
    avgSuccessRate: 0,
    avgConfidence: 0
    
  });
  
  // Enhanced filter state - added date range filter
  const [filters, setFilters] = useState({
    search: '',
    group: '',
    status: '',
    matchFilter: '', // 'match', 'no_match', or ''
    sortBy: 'lastActivity',
    sortOrder: 'desc',
    createdFrom: '',
    createdTo: ''
  });

  // Load data from API
  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Prepare filters for API call
      const apiFilters = {};
      if (filters.group) apiFilters.groupFilter = filters.group;
      if (filters.search) apiFilters.search = filters.search;
      if (filters.status) apiFilters.statusFilter = filters.status;

      const response = await lotSessionService.getDashboardData(apiFilters);
      
      if (response.success) {
        setDashboardData(response);
        setStatistics(lotSessionService.processStatistics(response));
        
        // Convert API data to component format
        const processedGroups = lotSessionService.processGroupedData(response);
        const allLots = Object.values(processedGroups).flatMap(group => 
          group.lots.map(lot => ({
            ...lot,
            completedSessions: lot.successfulSessions || 0,
            successfulMatches: lot.successfulSessions || 0,
            successRate: lot.sessionSuccessRate || 0,
            // Fix: Properly handle confidence values from API
            avgConfidence: lot.lotMatchConfidence ? parseFloat(lot.lotMatchConfidence).toFixed(1) : '0.0',
            createdAt: lot.createdAt,
            lastActivity: lot.lastActivity,
            // Determine lot-level status from sessions
            lotStatus: (lot.sessions || []).some(s => s.status === 'running') ? 'running' : 
                      (lot.sessions || []).some(s => s.status === 'pending') ? 'pending' :
                      (lot.sessions || []).every(s => s.status === 'completed') ? 'completed' : 'mixed',
            sessions: (lot.sessions || []).map(session => ({
              ...session,
              // Ensure confidence is properly formatted
              confidence: session.confidence ? parseFloat(session.confidence) : 0,
              timestamp: session.timestamp
            }))
          }))
        );
        
        setLots(allLots);
        setUseMockData(false);
      } else {
        throw new Error('Failed to load dashboard data');
      }
    } catch (err) {
      console.warn('Failed to load real data, using mock data:', err);
      setError('Failed to load real data. Using demo data.');
      setLots(generateMockData());
      setUseMockData(true);
    } finally {
      setLoading(false);
    }
  }, []); // Removed dependencies to prevent API calls on every filter change

  // Initial data load
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Get available groups for filter dropdown
  const availableGroups = useMemo(() => {
    return [...new Set(lots.map(lot => lot.group))].sort();
  }, [lots]);

  // Get available lot-level statuses for filter dropdown
  const availableStatuses = useMemo(() => {
    const statuses = new Set();
    lots.forEach(lot => {
      // Add the lot-level status
      statuses.add(lot.lotStatus || 'unknown');
    });
    return [...statuses].sort();
  }, [lots]);

  // Fixed filtering logic - added date range filtering
  const filteredLots = useMemo(() => {
    return lots.filter(lot => {
      // Search filter - search in lot name, lot ID, and expected piece
      if (filters.search) {
        const searchTerm = filters.search.toLowerCase();
        const matchesLotName = lot.lotName.toLowerCase().includes(searchTerm);
        const matchesExpectedPiece = lot.expectedPiece.toLowerCase().includes(searchTerm);
        const matchesLotId = lot.id.toLowerCase().includes(searchTerm);
        
        if (!matchesLotName && !matchesExpectedPiece && !matchesLotId) {
          return false;
        }
      }
      
      // Group filter - exact match on group name
      if (filters.group && lot.group !== filters.group) {
        return false;
      }
      
      // Status filter - check lot-level status
      if (filters.status && lot.lotStatus !== filters.status) {
        return false;
      }
      
      // Match filter - check if lot has matching or non-matching sessions
      if (filters.matchFilter) {
        const completedSessions = lot.sessions?.filter(session => session.status === 'completed') || [];
        const hasMatches = completedSessions.some(session => session.isTargetMatch);
        const hasNoMatches = completedSessions.some(session => !session.isTargetMatch);
        
        if (filters.matchFilter === 'match' && !hasMatches) {
          return false;
        }
        
        if (filters.matchFilter === 'no_match' && !hasNoMatches) {
          return false;
        }
      }
      
      // Date range filter - check created_at
      if (filters.createdFrom) {
        const createdFromDate = new Date(filters.createdFrom);
        const lotCreatedDate = new Date(lot.createdAt);
        if (lotCreatedDate < createdFromDate) {
          return false;
        }
      }
      
      if (filters.createdTo) {
        const createdToDate = new Date(filters.createdTo);
        createdToDate.setHours(23, 59, 59, 999); // End of day
        const lotCreatedDate = new Date(lot.createdAt);
        if (lotCreatedDate > createdToDate) {
          return false;
        }
      }
      
      return true;
    });
  }, [lots, filters]);

  // Enhanced group lots with proper filtering
  const groupedLots = useMemo(() => {
    const groups = {};
    
    filteredLots.forEach(lot => {
      if (!groups[lot.group]) {
        groups[lot.group] = {
          groupName: lot.group,
          lots: [],
          totalLots: 0,
          totalSessions: 0,
          avgSuccessRate: 0,
          avgConfidence: 0,
          lastActivity: null
        };
      }
      
      groups[lot.group].lots.push(lot);
      groups[lot.group].totalLots++;
      groups[lot.group].totalSessions += lot.totalSessions;
      
      if (!groups[lot.group].lastActivity || lot.lastActivity > groups[lot.group].lastActivity) {
        groups[lot.group].lastActivity = lot.lastActivity;
      }
    });

    // Calculate group averages
    Object.values(groups).forEach(group => {
      if (group.lots.length > 0) {
        group.avgSuccessRate = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / group.lots.length).toFixed(1);
        
        // FIXED: Include all lots in confidence calculation, not just those with >0 confidence
        group.avgConfidence = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / group.lots.length).toFixed(1);
      }
    });
    
    return groups;
  }, [filteredLots]);

  // FIXED: Update statistics based on filtered data with proper confidence and active groups calculation
  const finalStatistics = useMemo(() => {
    // FIXED: Calculate confidence for all lots, not just those with >0 confidence
    const avgConfidence = filteredLots.length > 0 
      ? (filteredLots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / filteredLots.length).toFixed(1)
      : '0.0';
      

      
    return {
      totalGroups: Object.keys(groupedLots).length,
      totalLots: filteredLots.length,
      totalSessions: filteredLots.reduce((acc, lot) => acc + lot.totalSessions, 0),
      avgSuccessRate: filteredLots.length > 0 ? (filteredLots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / filteredLots.length).toFixed(1) : 0,
      avgConfidence: avgConfidence,
     
    };
  }, [filteredLots, groupedLots]);

  const handleFilterChange = useCallback((field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
    setPage(0);
  }, []);

  const handleClearFilters = useCallback(() => {
    setFilters({
      search: '',
      group: '',
      status: '',
      matchFilter: '',
      sortBy: 'lastActivity',
      sortOrder: 'desc',
      createdFrom: '',
      createdTo: ''
    });
    setPage(0);
  }, []);

  const handleRefresh = useCallback(() => {
    if (lotSessionService?.clearCache) {
      lotSessionService.clearCache();
    }
    loadDashboardData();
  }, [loadDashboardData]);

  const handleLotExpand = useCallback((lotId) => {
    setExpandedLots(prev => {
      const newSet = new Set(prev);
      if (newSet.has(lotId)) {
        newSet.delete(lotId);
      } else {
        newSet.add(lotId);
      }
      return newSet;
    });
  }, []);

  const formatDate = useCallback((date) => {
    if (!date) return 'N/A';
    
    let dateObj;
    if (typeof date === 'string') {
      dateObj = new Date(date);
    } else if (date instanceof Date) {
      dateObj = date;
    } else {
      return 'Invalid Date';
    }
    
    if (isNaN(dateObj.getTime())) {
      return 'Invalid Date';
    }
    
    return dateObj.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }, []);

  const formatDateForInput = useCallback((date) => {
    if (!date) return '';
    const d = new Date(date);
    return d.toISOString().split('T')[0];
  }, []);

  const getSessionStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle fontSize="small" />;
      case 'failed': return <Error fontSize="small" />;
      case 'running': return <CircularProgress size={16} />;
      case 'pending': return <Schedule fontSize="small" />;
      default: return <Schedule fontSize="small" />;
    }
  };

  if (loading) {
    return (
      <Container>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <CircularProgress size={50} />
          <Typography variant="h6" sx={{ ml: 2 }}>Loading dashboard data...</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container>
      {/* Error Alert */}
      {error && (
        <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Header */}
      <HeaderBox>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, color: '#333', mb: 1 }}>
            Lot Session Database {useMockData && '(Demo Mode)'}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Monitor and analyze detection lot sessions grouped by production lines
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            startIcon={<FilterList />}
            onClick={() => setShowFilters(!showFilters)}
            variant={showFilters ? "contained" : "outlined"}
            sx={{ textTransform: "none" }}
          >
            Filters
          </Button>
          
          <Button
            startIcon={<Refresh />}
            onClick={handleRefresh}
            variant="outlined"
            sx={{ textTransform: "none" }}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </HeaderBox>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#667eea', fontWeight: 'bold', mb: 1 }}>
                {finalStatistics.totalGroups}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Production Groups
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>

        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#667eea', fontWeight: 'bold', mb: 1 }}>
                {finalStatistics.totalLots}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Lots
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>

        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#667eea', fontWeight: 'bold', mb: 1 }}>
                {finalStatistics.totalSessions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Sessions
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>

        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#4caf50', fontWeight: 'bold', mb: 1 }}>
                {finalStatistics.avgSuccessRate}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Success Rate
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>

        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#ff9800', fontWeight: 'bold', mb: 1 }}>
                {finalStatistics.avgConfidence}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Confidence
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>


      </Grid>

      {/* Enhanced Filters Panel with Date Range */}
      <Collapse in={showFilters}>
        <FilterCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: "#667eea" }}>
              Search & Filter Options
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Search"
                  value={filters.search}
                  onChange={(e) => handleFilterChange('search', e.target.value)}
                  placeholder="Search by lot name or piece..."
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Production Group</InputLabel>
                  <Select
                    value={filters.group}
                    onChange={(e) => handleFilterChange('group', e.target.value)}
                    label="Production Group"
                  >
                    <MenuItem value="">All Groups</MenuItem>
                    {availableGroups.map(group => (
                      <MenuItem key={group} value={group}>{group}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Lot Status</InputLabel>
                  <Select
                    value={filters.status}
                    onChange={(e) => handleFilterChange('status', e.target.value)}
                    label="Lot Status"
                  >
                    <MenuItem value="">All Statuses</MenuItem>
                    {availableStatuses.map(status => (
                      <MenuItem key={status} value={status}>
                        {status.charAt(0).toUpperCase() + status.slice(1)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Match Status</InputLabel>
                  <Select
                    value={filters.matchFilter}
                    onChange={(e) => handleFilterChange('matchFilter', e.target.value)}
                    label="Match Status"
                  >
                    <MenuItem value="">All Matches</MenuItem>
                    <MenuItem value="match">Has Matches</MenuItem>
                    <MenuItem value="no_match">Has No Matches</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Created From"
                  type="date"
                  value={filters.createdFrom}
                  onChange={(e) => handleFilterChange('createdFrom', e.target.value)}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <DateRange />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>

              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Created To"
                  type="date"
                  value={filters.createdTo}
                  onChange={(e) => handleFilterChange('createdTo', e.target.value)}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <DateRange />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                  <Button
                    startIcon={<Clear />}
                    onClick={handleClearFilters}
                    variant="outlined"
                    sx={{ textTransform: "none" }}
                  >
                    Clear Filters
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </FilterCard>
      </Collapse>

      {/* Groups and Lots */}
      {Object.entries(groupedLots).map(([groupName, group]) => (
        <GroupAccordion key={groupName} defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
              <Avatar sx={{ bgcolor: '#667eea', color: 'white', width: 40, height: 40 }}>
                <GroupIcon />
              </Avatar>
              
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h6" sx={{ fontWeight: 600, color: '#333' }}>
                  {groupName}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {group.totalLots} lots • {group.totalSessions} sessions • Last activity: {formatDate(group.lastActivity)}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', gap: 3, mr: 2 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">Success Rate</Typography>
                  <Typography variant="body1" fontWeight="600" color="#4caf50">
                    {group.avgSuccessRate}%
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">Avg Confidence</Typography>
                  <Typography variant="body1" fontWeight="600" color="#ff9800">
                    {group.avgConfidence}%
                  </Typography>
                </Box>
              </Box>
            </Box>
          </AccordionSummary>
          
          <AccordionDetails>
            {group.lots.map((lot) => (
              <Card key={lot.id} sx={{ mb: 2, border: '1px solid rgba(102, 126, 234, 0.1)' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: '#333' }}>
                        {lot.lotName}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Expected: {lot.expectedPiece} (#{lot.expectedPieceNumber}) • Created: {formatDate(lot.createdAt)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                      <Chip 
                        label={`${lot.totalSessions} Sessions`} 
                        size="small" 
                        sx={{ bgcolor: '#f3f4f6', color: '#374151' }}
                      />
                      <Chip 
                        label={`${lot.successRate}% Success`} 
                        size="small" 
                        sx={{ bgcolor: '#e8f5e8', color: '#2e7d2e' }}
                      />
                      <Button
                        size="small"
                        onClick={() => handleLotExpand(lot.id)}
                        endIcon={<Visibility />}
                        sx={{ textTransform: 'none' }}
                      >
                        {expandedLots.has(lot.id) ? 'Hide' : 'Show'} Sessions
                      </Button>
                    </Box>
                  </Box>

                  {/* Progress Bars */}
                  <Box sx={{ display: 'flex', gap: 4, mb: 2 }}>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Completion Rate ({lot.completedSessions}/{lot.totalSessions})
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(lot.completedSessions / lot.totalSessions) * 100}
                        sx={{ 
                          height: 6, 
                          borderRadius: 3,
                          bgcolor: 'rgba(102, 126, 234, 0.1)',
                          '& .MuiLinearProgress-bar': { bgcolor: '#667eea' }
                        }}
                      />
                    </Box>
                    
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Success Rate ({lot.successfulMatches}/{lot.completedSessions})
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={lot.completedSessions > 0 ? (lot.successfulMatches / lot.completedSessions) * 100 : 0}
                        sx={{ 
                          height: 6, 
                          borderRadius: 3,
                          bgcolor: 'rgba(76, 175, 80, 0.1)',
                          '& .MuiLinearProgress-bar': { bgcolor: '#4caf50' }
                        }}
                      />
                    </Box>
                  </Box>

                  {/* Sessions Table */}
                  <Collapse in={expandedLots.has(lot.id)}>
                    <TableContainer component={Paper} sx={{ mt: 2, border: '1px solid rgba(102, 126, 234, 0.1)' }}>
                      <SessionTable size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Session</TableCell>
                            <TableCell>Camera</TableCell>
                            <TableCell>Target Piece</TableCell>
                            <TableCell>Detected</TableCell>
                            <TableCell>Match</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Time</TableCell>
                            <TableCell>Timestamp</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {lot.sessions.map((session) => (
                            <TableRow key={session.id}>
                              <TableCell>
                                <Typography variant="body2" fontWeight="600">
                                  #{session.sessionNumber}
                                </Typography>
                              </TableCell>
                              
                              <TableCell>
                                <Chip
                                  icon={<Camera />}
                                  label={`Cam ${session.cameraId}`}
                                  size="small"
                                  sx={{ bgcolor: '#e3f2fd', color: '#1976d2' }}
                                />
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="body2" fontWeight="500">
                                  {session.targetPiece}
                                </Typography>
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="body2" color={session.detectedPiece === session.targetPiece ? '#4caf50' : '#f44336'}>
                                  {session.detectedPiece}
                                </Typography>
                              </TableCell>
                              
                              <TableCell>
                                <Chip
                                  icon={session.isTargetMatch ? <CheckCircle /> : <Error />}
                                  label={session.isTargetMatch ? 'Match' : 'No Match'}
                                  size="small"
                                  variant={session.isTargetMatch ? 'completed' : 'failed'}
                                  sx={{
                                    bgcolor: session.isTargetMatch ? '#e8f5e8' : '#ffebee',
                                    color: session.isTargetMatch ? '#2e7d2e' : '#c62828',
                                  }}
                                />
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="body2" fontWeight="600" color="#ff9800">
                                  {(session.confidence || 0).toFixed(1)}%
                                </Typography>
                              </TableCell>
                              
                              <TableCell>
                                <StatusChip
                                  icon={getSessionStatusIcon(session.status)}
                                  label={session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                                  size="small"
                                  variant={session.status}
                                />
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="body2" color="text.secondary">
                                  {session.processingTime}ms
                                </Typography>
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="caption" color="text.secondary">
                                  {formatDate(session.timestamp)}
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </SessionTable>
                    </TableContainer>
                  </Collapse>
                </CardContent>
              </Card>
            ))}
          </AccordionDetails>
        </GroupAccordion>
      ))}

      {/* Empty State */}
      {Object.keys(groupedLots).length === 0 && (
        <Card sx={{ textAlign: 'center', py: 8, border: '1px solid rgba(102, 126, 234, 0.1)' }}>
          <CardContent>
            <Analytics sx={{ fontSize: 64, color: '#ccc', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No lots found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Try adjusting your filters or refresh the data
            </Typography>
            <Button
              variant="outlined"
              onClick={handleClearFilters}
              startIcon={<Clear />}
              sx={{ textTransform: 'none' }}
            >
              Clear Filters
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Pagination */}
      <TablePagination
        component="div"
        count={Object.values(groupedLots).reduce((acc, group) => acc + group.totalLots, 0)}
        page={page}
        onPageChange={(e, newPage) => setPage(newPage)}
        rowsPerPage={pageSize}
        onRowsPerPageChange={(e) => {
          setPageSize(parseInt(e.target.value, 10));
          setPage(0);
        }}
        rowsPerPageOptions={[5, 10, 25, 50]}
        sx={{
          mt: 3,
          bgcolor: 'rgba(102, 126, 234, 0.02)',
          borderRadius: '12px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          "& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows": {
            color: "#667eea",
            fontWeight: "500"
          }
        }}
      />
    </Container>
  );
}