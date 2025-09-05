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
  Badge,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider
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
  Visibility,
  DateRange,
  ViewList,
  Close,
  PrecisionManufacturing,
  PhotoCamera,
  Timeline,
  BoundingBox
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Styled Components
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

const DetectedPieceCard = styled(Card)(({ theme }) => ({
  marginBottom: '8px',
  border: '1px solid rgba(0, 0, 0, 0.1)',
  borderRadius: '8px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
}));

// API Service - simplified version focusing on basic fetching
const apiService = {
  async fetchDashboardData(filters = {}) {
    try {
      const params = new URLSearchParams();
      if (filters.groupFilter) params.append('group_filter', filters.groupFilter);
      if (filters.search) params.append('search', filters.search);
      if (filters.statusFilter) params.append('status_filter', filters.statusFilter);

      const url = `/api/detection/basic/lotSession/data${params.toString() ? `?${params.toString()}` : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API fetch failed:', error);
      throw error;
    }
  },

  async fetchLotSessions(lotId) {
    try {
      const response = await fetch(`/api/detection/basic/lots/${lotId}/sessions`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Fetch lot sessions failed:', error);
      throw error;
    }
  },

  async fetchSessionDetectedPieces(sessionId) {
    try {
      const response = await fetch(`/api/detection/basic/lotSession/sessions/${sessionId}/pieces`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Fetch detected pieces failed:', error);
      throw error;
    }
  }
};

// DetectedPiecesDialog Component
const DetectedPiecesDialog = ({ open, onClose, session, detectedPieces, loading }) => {
  const correctPieces = detectedPieces.filter(piece => piece.is_correct_piece);
  const incorrectPieces = detectedPieces.filter(piece => !piece.is_correct_piece);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">
              Detected Pieces - Session #{session?.sessionNumber}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {session?.targetPiece} • {detectedPieces.length} pieces detected
            </Typography>
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Box>
            {/* Summary Stats */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={3}>
                <Card sx={{ textAlign: 'center', py: 2, bgcolor: '#e8f5e8' }}>
                  <Typography variant="h4" color="#2e7d2e">{correctPieces.length}</Typography>
                  <Typography variant="caption">Correct</Typography>
                </Card>
              </Grid>
              <Grid item xs={3}>
                <Card sx={{ textAlign: 'center', py: 2, bgcolor: '#ffebee' }}>
                  <Typography variant="h4" color="#c62828">{incorrectPieces.length}</Typography>
                  <Typography variant="caption">Incorrect</Typography>
                </Card>
              </Grid>
              <Grid item xs={3}>
                <Card sx={{ textAlign: 'center', py: 2, bgcolor: '#f3f4f6' }}>
                  <Typography variant="h4" color="#374151">{detectedPieces.length}</Typography>
                  <Typography variant="caption">Total</Typography>
                </Card>
              </Grid>
              {/* <Grid item xs={3}>
                <Card sx={{ textAlign: 'center', py: 2, bgcolor: '#fff3e0' }}>
                  <Typography variant="h4" color="#ef6c00">
                    {detectedPieces.length > 0 
                      ? (detectedPieces.reduce((acc, p) => acc + p.confidence_score, 0) / detectedPieces.length).toFixed(1)
                      : 0}%
                  </Typography>
                  <Typography variant="caption">Avg Confidence</Typography>
                </Card>
              </Grid> */}
            </Grid>

            <Divider sx={{ my: 2 }} />

            {/* Correct Pieces */}
            {correctPieces.length > 0 && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" sx={{ color: '#2e7d2e', mb: 2, display: 'flex', alignItems: 'center' }}>
                  <CheckCircle sx={{ mr: 1 }} />
                  Correct Pieces ({correctPieces.length})
                </Typography>
                {correctPieces.map((piece, index) => (
                  <DetectedPieceCard key={piece.id}>
                    <CardContent sx={{ py: 2 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={3}>
                          <Typography variant="body2" fontWeight="600">
                            {piece.detected_label}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Piece ID: {piece.piece_id || 'Unknown'}
                          </Typography>
                        </Grid>
                        <Grid item xs={2}>
                          <Typography variant="body2" color="#2e7d2e" fontWeight="600">
                            {(piece.confidence_score * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Confidence
                          </Typography>
                        </Grid>
                        <Grid item xs={4}>
                          <Typography variant="caption" color="text.secondary">
                            Bounding Box: ({piece.bounding_box_x1}, {piece.bounding_box_y1}) - ({piece.bounding_box_x2}, {piece.bounding_box_y2})
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(piece.created_at).toLocaleTimeString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </DetectedPieceCard>
                ))}
              </Box>
            )}

            {/* Incorrect Pieces */}
            {incorrectPieces.length > 0 && (
              <Box>
                <Typography variant="h6" sx={{ color: '#c62828', mb: 2, display: 'flex', alignItems: 'center' }}>
                  <Error sx={{ mr: 1 }} />
                  Incorrect Pieces ({incorrectPieces.length})
                </Typography>
                {incorrectPieces.map((piece, index) => (
                  <DetectedPieceCard key={piece.id}>
                    <CardContent sx={{ py: 2 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={3}>
                          <Typography variant="body2" fontWeight="600">
                            {piece.detected_label}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Piece ID: {piece.piece_id || 'Unknown'}
                          </Typography>
                        </Grid>
                        <Grid item xs={2}>
                          <Typography variant="body2" color="#c62828" fontWeight="600">
                            {(piece.confidence_score * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Confidence
                          </Typography>
                        </Grid>
                        <Grid item xs={4}>
                          <Typography variant="caption" color="text.secondary">
                            Bounding Box: ({piece.bounding_box_x1}, {piece.bounding_box_y1}) - ({piece.bounding_box_x2}, {piece.bounding_box_y2})
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(piece.created_at).toLocaleTimeString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </DetectedPieceCard>
                ))}
              </Box>
            )}

            {detectedPieces.length === 0 && (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <PrecisionManufacturing sx={{ fontSize: 48, color: '#ccc', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No detected pieces found for this session
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

// Main Component
export default function LotSessionDatabase() {
  const [lots, setLots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  const [expandedLots, setExpandedLots] = useState(new Set());
  const [dashboardData, setDashboardData] = useState(null);
  const [detectedPiecesDialog, setDetectedPiecesDialog] = useState({
    open: false,
    session: null,
    pieces: [],
    loading: false
  });
  
  const [filters, setFilters] = useState({
    search: '',
    group: '',
    status: '',
    matchFilter: '',
    sortBy: 'lastActivity',
    sortOrder: 'desc',
    createdFrom: '',
    createdTo: ''
  });

  const [statistics, setStatistics] = useState({
    totalGroups: 0,
    totalLots: 0,
    totalSessions: 0,
    avgSuccessRate: 0,
    avgConfidence: 0
  });

  // Load dashboard data from API
  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const apiFilters = {};
      if (filters.group) apiFilters.groupFilter = filters.group;
      if (filters.search) apiFilters.search = filters.search;
      if (filters.status) apiFilters.statusFilter = filters.status;

      const response = await apiService.fetchDashboardData(apiFilters);
      
      if (response.success) {
        setDashboardData(response);
        
        // Process grouped lots from API response
        const processedLots = [];
        
        Object.entries(response.groupedLots || {}).forEach(([groupName, groupLots]) => {
          groupLots.forEach(lot => {
            const processedLot = {
              id: lot.id,
              group: groupName,
              lotName: lot.lotName,
              expectedPiece: lot.expectedPiece,
              expectedPieceNumber: lot.expectedPieceNumber,
              sessions: lot.sessions || [],
              totalSessions: lot.totalSessions || 0,
              completedSessions: lot.completedSessions || 0,
              successfulMatches: lot.successfulSessions || 0,
              successRate: parseFloat(lot.sessionSuccessRate || 0).toFixed(1),
              avgConfidence: parseFloat(lot.avgConfidence || 0).toFixed(1),
              createdAt: lot.createdAt,
              lastActivity: lot.lastActivity,
              lotStatus: determineLotStatus(lot.sessions || [])
            };
            processedLots.push(processedLot);
          });
        });
        
        setLots(processedLots);
        
        // Set statistics from API response
        setStatistics({
          totalGroups: response.statistics?.totalGroups || 0,
          totalLots: response.statistics?.totalLots || 0,
          totalSessions: response.statistics?.totalSessions || 0,
          avgSuccessRate: parseFloat(response.statistics?.sessionSuccessRate || 0).toFixed(1),
          avgConfidence: parseFloat(response.statistics?.avgLotConfidence || 0).toFixed(1)
        });
      } else {
        throw new Error(response.message || 'Failed to load dashboard data');
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(`Failed to load data: ${err.message}`);
      setLots([]);
    } finally {
      setLoading(false);
    }
  }, [filters.group, filters.search, filters.status]);

  // Determine lot status based on sessions
  const determineLotStatus = (sessions) => {
    if (!sessions || sessions.length === 0) return 'pending';
    
    const hasRunning = sessions.some(s => s.status === 'running');
    const hasPending = sessions.some(s => s.status === 'pending');
    const allCompleted = sessions.every(s => s.status === 'completed');
    
    if (hasRunning) return 'running';
    if (hasPending) return 'pending';
    if (allCompleted) return 'completed';
    return 'mixed';
  };

  // Initial data load
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Handle detected pieces dialog
  const handleShowDetectedPieces = async (session) => {
    setDetectedPiecesDialog({
      open: true,
      session: session,
      pieces: [],
      loading: true
    });

    try {
      // First check if pieces are already in session data
      if (session.detected_pieces && session.detected_pieces.length > 0) {
        setDetectedPiecesDialog(prev => ({
          ...prev,
          pieces: session.detected_pieces,
          loading: false
        }));
        return;
      }

      // Otherwise fetch from API
      const response = await apiService.fetchSessionDetectedPieces(session.id);
      
      setDetectedPiecesDialog(prev => ({
        ...prev,
        pieces: response.detected_pieces || [],
        loading: false
      }));
    } catch (error) {
      console.error('Error loading detected pieces:', error);
      setDetectedPiecesDialog(prev => ({
        ...prev,
        pieces: [],
        loading: false
      }));
    }
  };

  // Get available groups for filtering
  const availableGroups = useMemo(() => {
    return [...new Set(lots.map(lot => lot.group))].sort();
  }, [lots]);

  const availableStatuses = useMemo(() => {
    const statuses = new Set();
    lots.forEach(lot => {
      statuses.add(lot.lotStatus || 'unknown');
    });
    return [...statuses].sort();
  }, [lots]);

  // Apply filters
  const filteredLots = useMemo(() => {
    return lots.filter(lot => {
      if (filters.search) {
        const searchTerm = filters.search.toLowerCase();
        const matchesLotName = lot.lotName.toLowerCase().includes(searchTerm);
        const matchesExpectedPiece = lot.expectedPiece.toLowerCase().includes(searchTerm);
        const matchesLotId = lot.id.toString().toLowerCase().includes(searchTerm);
        
        if (!matchesLotName && !matchesExpectedPiece && !matchesLotId) {
          return false;
        }
      }
      
      if (filters.group && lot.group !== filters.group) {
        return false;
      }
      
      if (filters.status && lot.lotStatus !== filters.status) {
        return false;
      }
      
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
      
      if (filters.createdFrom) {
        const createdFromDate = new Date(filters.createdFrom);
        const lotCreatedDate = new Date(lot.createdAt);
        if (lotCreatedDate < createdFromDate) {
          return false;
        }
      }
      
      if (filters.createdTo) {
        const createdToDate = new Date(filters.createdTo);
        createdToDate.setHours(23, 59, 59, 999);
        const lotCreatedDate = new Date(lot.createdAt);
        if (lotCreatedDate > createdToDate) {
          return false;
        }
      }
      
      return true;
    });
  }, [lots, filters]);

  // Group filtered lots
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
      
      if (!groups[lot.group].lastActivity || new Date(lot.lastActivity) > new Date(groups[lot.group].lastActivity)) {
        groups[lot.group].lastActivity = lot.lastActivity;
      }
    });

    // Calculate group averages
    Object.values(groups).forEach(group => {
      if (group.lots.length > 0) {
        group.avgSuccessRate = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / group.lots.length).toFixed(1);
        group.avgConfidence = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / group.lots.length).toFixed(1);
      }
    });
    
    return groups;
  }, [filteredLots]);

  // Update statistics based on filtered data
  const finalStatistics = useMemo(() => {
    const avgConfidence = filteredLots.length > 0 
      ? (filteredLots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / filteredLots.length).toFixed(1)
      : '0.0';
      
    return {
      totalGroups: Object.keys(groupedLots).length,
      totalLots: filteredLots.length,
      totalSessions: filteredLots.reduce((acc, lot) => acc + lot.totalSessions, 0),
      avgSuccessRate: filteredLots.length > 0 ? (filteredLots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / filteredLots.length).toFixed(1) : 0,
      avgConfidence: avgConfidence
    };
  }, [filteredLots, groupedLots]);

  // Event handlers
  const handleFilterChange = useCallback((field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
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
  }, []);

  const handleRefresh = useCallback(() => {
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
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Header */}
      <HeaderBox>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, color: '#333', mb: 1 }}>
            Lot Session Database
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Monitor and analyze detection lot sessions with detailed piece tracking
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

      {/* Filters Panel */}
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
                            <TableCell>Target Piece</TableCell>
                            <TableCell>Match</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Detected Pieces</TableCell>
                            <TableCell>Timestamp</TableCell>
                            <TableCell>Actions</TableCell>
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
                                <Typography variant="body2" fontWeight="500">
                                  {session.targetPiece}
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
                                  {((session.confidence || 0)*100).toFixed(1)}%
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
                                <Badge 
                                  badgeContent={session.detected_pieces?.length || session.detectedPieces?.length || 0} 
                                  color="primary"
                                  max={99}
                                >
                                  <PrecisionManufacturing />
                                </Badge>
                              </TableCell>
                              
                              <TableCell>
                                <Typography variant="caption" color="text.secondary">
                                  {formatDate(session.timestamp)}
                                </Typography>
                              </TableCell>

                              <TableCell>
                                <Tooltip title="View Detected Pieces">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleShowDetectedPieces(session)}
                                    disabled={!session.detected_pieces?.length && !session.detectedPieces?.length}
                                  >
                                    <ViewList />
                                  </IconButton>
                                </Tooltip>
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
      {Object.keys(groupedLots).length === 0 && !loading && (
        <Card sx={{ textAlign: 'center', py: 8, border: '1px solid rgba(102, 126, 234, 0.1)' }}>
          <CardContent>
            <Analytics sx={{ fontSize: 64, color: '#ccc', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No lots found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {error ? 'Failed to load data from the server.' : 'Try adjusting your filters or refresh the data'}
            </Typography>
            <Button
              variant="outlined"
              onClick={error ? handleRefresh : handleClearFilters}
              startIcon={error ? <Refresh /> : <Clear />}
              sx={{ textTransform: 'none' }}
            >
              {error ? 'Retry' : 'Clear Filters'}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Detected Pieces Dialog */}
      <DetectedPiecesDialog
        open={detectedPiecesDialog.open}
        onClose={() => setDetectedPiecesDialog({ open: false, session: null, pieces: [], loading: false })}
        session={detectedPiecesDialog.session}
        detectedPieces={detectedPiecesDialog.pieces}
        loading={detectedPiecesDialog.loading}
      />
    </Container>
  );
}