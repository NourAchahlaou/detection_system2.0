import React from 'react';
import {
  Box,
  Button,
  CircularProgress,
} from "@mui/material";
import {
  FilterList,
  Refresh,
  PlayArrow,
  Stop,
  ModelTraining,
} from "@mui/icons-material";
import { HeaderBox, TrainButton } from './StyledComponents';

export default function HeaderActions({
  onTrainAll,
  onStopTraining,
  onToggleSidebar,
  onToggleFilters, // This prop name should match what's passed from parent
  onRefresh,
  onBulkDelete,
  trainingInProgress,
  sidebarOpen,
  showFilters,
  selectedCount
}) {
  return (
    <HeaderBox>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <TrainButton
          variant="contained"
          color="primary"
          onClick={onTrainAll}
          disabled={trainingInProgress}
          startIcon={trainingInProgress ? <CircularProgress size={16} color="inherit" /> : <PlayArrow />}
        >
          {trainingInProgress ? "Training..." : "Train All"}
        </TrainButton>
        
        {trainingInProgress && (
          <TrainButton
            variant="outlined"
            color="error"
            onClick={onStopTraining}
            startIcon={<Stop />}
          >
            Stop Training
          </TrainButton>
        )}
        
        {trainingInProgress && (
          <Button
            variant="outlined"
            onClick={onToggleSidebar}
            startIcon={<ModelTraining />}
            sx={{ textTransform: "none" }}
          >
            {sidebarOpen ? "Hide Progress" : "Show Progress"}
          </Button>
        )}
        
        <Button
          startIcon={<FilterList />}
          onClick={onToggleFilters} // Make sure this matches the prop name
          variant={showFilters ? "contained" : "outlined"}
          sx={{ textTransform: "none" }}
        >
          Filters
        </Button>
        
        <Button
          startIcon={<Refresh />}
          onClick={onRefresh}
          variant="outlined"
          sx={{ textTransform: "none" }}
        >
          Refresh
        </Button>
        
        {selectedCount > 0 && (
          <Button
            variant="contained"
            color="error"
            onClick={onBulkDelete}
            sx={{ textTransform: "none" }}
          >
            Delete Selected ({selectedCount})
          </Button>
        )}
      </Box>
    </HeaderBox>
  );
}