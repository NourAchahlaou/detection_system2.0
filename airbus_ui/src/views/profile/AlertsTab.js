import * as React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  FormControl, 
  Select, 
  MenuItem, 
  List, 
  Chip, 
  Divider 
} from '@mui/material';

export default function AlertsTab() {
  const [filterStatus, setFilterStatus] = React.useState('All');
  const [filterType, setFilterType] = React.useState('All');
  
  const alertsData = [
    { 
      id: "ALT-1523", 
      type: "Critical", 
      message: "Wrong lot detected for part D532.31954.010.10", 
      time: "14:05", 
      date: "07/05/2025",
      status: "Unresolved" 
    },
    { 
      id: "ALT-1522", 
      type: "Warning", 
      message: "Detection confidence below threshold (78%) for part D532.31953.015.10", 
      time: "13:42", 
      date: "07/05/2025",
      status: "Unresolved" 
    },
    { 
      id: "ALT-1521", 
      type: "Critical", 
      message: "Defect detected on part D532.31953.012.10", 
      time: "11:23", 
      date: "07/05/2025",
      status: "Unresolved" 
    },
    { 
      id: "ALT-1520", 
      type: "Info", 
      message: "System maintenance scheduled for 08/05/2025 22:00", 
      time: "09:30", 
      date: "07/05/2025",
      status: "Acknowledged" 
    },
    { 
      id: "ALT-1519", 
      type: "Warning", 
      message: "Camera 2 calibration may need adjustment", 
      time: "16:15", 
      date: "06/05/2025",
      status: "Resolved" 
    },
  ];
  
  const getChipColor = (type) => {
    switch (type) {
      case 'Critical':
        return { bg: '#FDE8E8', color: '#E53E3E' };
      case 'Warning':
        return { bg: '#FEFCBF', color: '#D69E2E' };
      case 'Info':
        return { bg: '#E6FFFA', color: '#38B2AC' };
      default:
        return { bg: '#EDF2F7', color: '#4A5568' };
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'Unresolved':
        return { bg: '#FED7D7', color: '#C53030' };
      case 'Acknowledged':
        return { bg: '#BEE3F8', color: '#2B6CB0' };
      case 'Resolved':
        return { bg: '#C6F6D5', color: '#2F855A' };
      default:
        return { bg: '#EDF2F7', color: '#4A5568' };
    }
  };
  
  const filteredAlerts = alertsData.filter(alert => {
    const matchesStatus = filterStatus === 'All' || alert.status === filterStatus;
    const matchesType = filterType === 'All' || alert.type === filterType;
    return matchesStatus && matchesType;
  });
  
  const handleStatusChange = (event) => {
    setFilterStatus(event.target.value);
  };
  
  const handleTypeChange = (event) => {
    setFilterType(event.target.value);
  };
  
  const handleClearFilters = () => {
    setFilterStatus('All');
    setFilterType('All');
  };
  
  const handleAcknowledge = (id) => {
    console.log(`Acknowledged alert ${id}`);
    // In a real app, you would update the alert status here
  };
  
  const handleResolve = (id) => {
    console.log(`Resolved alert ${id}`);
    // In a real app, you would update the alert status here
  };
  
  return (
    <Box>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
        <Typography variant="subtitle1" fontWeight="medium">Alerts & Notifications</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={filterStatus}
              onChange={handleStatusChange}
              displayEmpty
              variant="outlined"
            >
              <MenuItem value="All">All Status</MenuItem>
              <MenuItem value="Unresolved">Unresolved</MenuItem>
              <MenuItem value="Acknowledged">Acknowledged</MenuItem>
              <MenuItem value="Resolved">Resolved</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={filterType}
              onChange={handleTypeChange}
              displayEmpty
              variant="outlined"
            >
              <MenuItem value="All">All Types</MenuItem>
              <MenuItem value="Critical">Critical</MenuItem>
              <MenuItem value="Warning">Warning</MenuItem>
              <MenuItem value="Info">Info</MenuItem>
            </Select>
          </FormControl>
          
          <Button 
            variant="outlined" 
            size="small"
            onClick={handleClearFilters}
            disabled={filterStatus === 'All' && filterType === 'All'}
          >
            Clear Filters
          </Button>
        </Box>
      </Box>
      
      <Divider />
      
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Showing {filteredAlerts.length} of {alertsData.length} alerts
        </Typography>
        
        <List sx={{ width: '100%' }}>
          {filteredAlerts.length > 0 ? (
            filteredAlerts.map((alert, index) => (
              <React.Fragment key={alert.id}>
                <Box sx={{ py: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle2" fontWeight="medium">
                        {alert.id}
                      </Typography>
                      <Chip 
                        label={alert.type} 
                        size="small"
                        sx={{ 
                          backgroundColor: getChipColor(alert.type).bg,
                          color: getChipColor(alert.type).color,
                          fontWeight: 'medium'
                        }}
                      />
                      <Chip 
                        label={alert.status} 
                        size="small"
                        sx={{ 
                          backgroundColor: getStatusColor(alert.status).bg,
                          color: getStatusColor(alert.status).color,
                          fontWeight: 'medium'
                        }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {alert.time} • {alert.date}
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    {alert.message}
                  </Typography>
                  
                  {alert.status !== 'Resolved' && (
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                      {alert.status === 'Unresolved' && (
                        <Button 
                          size="small" 
                          variant="outlined"
                          onClick={() => handleAcknowledge(alert.id)}
                        >
                          Acknowledge
                        </Button>
                      )}
                      <Button 
                        size="small" 
                        variant="contained"
                        onClick={() => handleResolve(alert.id)}
                      >
                        Resolve
                      </Button>
                    </Box>
                  )}
                </Box>
                {index < filteredAlerts.length - 1 && <Divider />}
              </React.Fragment>
            ))
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body1" color="text.secondary">
                No alerts match your current filters
              </Typography>
              <Button 
                variant="text" 
                onClick={handleClearFilters}
                sx={{ mt: 1 }}
              >
                Clear filters
              </Button>
            </Box>
          )}
        </List>
      </Box>
    </Box>
  );
}