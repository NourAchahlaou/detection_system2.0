import React, { useState, useMemo, useCallback } from 'react';
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
  CircularProgress
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
  Visibility
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

// Mock data generator
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
          confidence: confidence,
          isTargetMatch: isMatch,
          status: status,
          timestamp: new Date(Date.now() - Math.random() * 86400000 * 7), // Last 7 days
          processingTime: Math.floor(Math.random() * 2000) + 500, // 500-2500ms
          detectionRate: Math.random() * 0.4 + 0.6 // 0.6-1.0
        });
      }
      
      const completedSessions = sessions.filter(s => s.status === 'completed').length;
      const matchingSessions = sessions.filter(s => s.isTargetMatch && s.status === 'completed').length;
      
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
        avgConfidence: sessions.filter(s => s.status === 'completed').length > 0 
          ? (sessions.filter(s => s.status === 'completed').reduce((acc, s) => acc + s.confidence, 0) / sessions.filter(s => s.status === 'completed').length * 100).toFixed(1)
          : 0,
        createdAt: new Date(Date.now() - Math.random() * 86400000 * 30), // Last 30 days
        lastActivity: sessions.length > 0 ? new Date(Math.max(...sessions.map(s => s.timestamp.getTime()))) : new Date()
      });
    }
  });
  
  return lots;
};

export default function LotSessionDatabase() {
  const [lots] = useState(generateMockData());
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [expandedLots, setExpandedLots] = useState(new Set());
  
  // Filter state
  const [filters, setFilters] = useState({
    search: '',
    group: '',
    status: '',
    sortBy: 'lastActivity',
    sortOrder: 'desc',
    dateFrom: '',
    dateTo: '',
    minSessions: '',
    maxSessions: ''
  });

  // Group lots by group
  const groupedLots = useMemo(() => {
    const groups = {};
    
    // Filter lots first
    const filteredLots = lots.filter(lot => {
      if (filters.search && !lot.lotName.toLowerCase().includes(filters.search.toLowerCase()) && 
          !lot.expectedPiece.toLowerCase().includes(filters.search.toLowerCase())) {
        return false;
      }
      
      if (filters.group && lot.group !== filters.group) {
        return false;
      }
      
      if (filters.minSessions && lot.totalSessions < parseInt(filters.minSessions)) {
        return false;
      }
      
      if (filters.maxSessions && lot.totalSessions > parseInt(filters.maxSessions)) {
        return false;
      }
      
      return true;
    });

    // Group filtered lots
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

    // Calculate averages for each group
    Object.values(groups).forEach(group => {
      if (group.lots.length > 0) {
        group.avgSuccessRate = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / group.lots.length).toFixed(1);
        group.avgConfidence = (group.lots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / group.lots.length).toFixed(1);
      }
    });
    
    return groups;
  }, [lots, filters]);

  // Statistics
  const statistics = useMemo(() => {
    return {
      totalGroups: Object.keys(groupedLots).length,
      totalLots: lots.length,
      totalSessions: lots.reduce((acc, lot) => acc + lot.totalSessions, 0),
      avgSuccessRate: lots.length > 0 ? (lots.reduce((acc, lot) => acc + parseFloat(lot.successRate), 0) / lots.length).toFixed(1) : 0,
      avgConfidence: lots.length > 0 ? (lots.reduce((acc, lot) => acc + parseFloat(lot.avgConfidence), 0) / lots.length).toFixed(1) : 0,
      activeGroups: Object.values(groupedLots).filter(group => 
        group.lots.some(lot => lot.sessions.some(session => session.status === 'running'))
      ).length
    };
  }, [lots, groupedLots]);

  const handleFilterChange = useCallback((field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
    setPage(0);
  }, []);

  const handleClearFilters = useCallback(() => {
    setFilters({
      search: '',
      group: '',
      status: '',
      sortBy: 'lastActivity',
      sortOrder: 'desc',
      dateFrom: '',
      dateTo: '',
      minSessions: '',
      maxSessions: ''
    });
    setPage(0);
  }, []);

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
    return date.toLocaleDateString('en-US', {
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

  return (
    <Container>
      {/* Header */}
      <HeaderBox>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, color: '#333', mb: 1 }}>
            Lot Session Database
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
            onClick={() => setLoading(!loading)}
            variant="outlined"
            sx={{ textTransform: "none" }}
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
                {statistics.totalGroups}
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
                {statistics.totalLots}
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
                {statistics.totalSessions}
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
                {statistics.avgSuccessRate}%
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
                {statistics.avgConfidence}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Confidence
              </Typography>
            </CardContent>
          </StatsCard>
        </Grid>

        <Grid item xs={12} sm={6} md={2}>
          <StatsCard>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="h3" sx={{ color: '#f44336', fontWeight: 'bold', mb: 1 }}>
                {statistics.activeGroups}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active Groups
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
              <Grid item xs={12} md={4}>
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
              
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Production Group</InputLabel>
                  <Select
                    value={filters.group}
                    onChange={(e) => handleFilterChange('group', e.target.value)}
                    label="Production Group"
                  >
                    <MenuItem value="">All Groups</MenuItem>
                    {Object.keys(groupedLots).map(group => (
                      <MenuItem key={group} value={group}>{group}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Min Sessions"
                  type="number"
                  value={filters.minSessions}
                  onChange={(e) => handleFilterChange('minSessions', e.target.value)}
                  placeholder="Minimum session count"
                />
              </Grid>

              <Grid item xs={12}>
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
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Camera fontSize="small" color="action" />
                                  <Typography variant="body2">
                                    Camera {session.cameraId}
                                  </Typography>
                                </Box>
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
                                  {(session.confidence * 100).toFixed(1)}%
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
          bgcolor: 'white',
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