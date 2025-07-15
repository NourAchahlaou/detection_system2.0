import React from 'react';
import {
  Box,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Button,
  Collapse,
  InputAdornment,
} from "@mui/material";
import { Search, Clear } from "@mui/icons-material";
import { FilterCard } from './StyledComponents';

export default function FiltersPanel({ 
  showFilters, 
  filters, 
  availableGroups, 
  onFilterChange, 
  onClearFilters 
}) {
  return (
    <Collapse in={showFilters}>
      <FilterCard>
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: "#667eea" }}>
            Search & Filter Options
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Search"
                value={filters.search}
                onChange={(e) => onFilterChange('search', e.target.value)}
                placeholder="Search by label or class ID..."
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Group</InputLabel>
                <Select
                  value={filters.group_filter}
                  onChange={(e) => onFilterChange('group_filter', e.target.value)}
                  label="Group"
                  sx={{
                    '& .MuiSelect-select': {
                      whiteSpace: 'nowrap',
                      overflow: 'visible',
                      textOverflow: 'unset',
                    },
                  }}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 300,
                        width: 'auto',
                        minWidth: 200,
                      },
                    },
                  }}
                >
                  <MenuItem value="">All Groups</MenuItem>
                  {availableGroups.map(group => (
                    <MenuItem key={group} value={group}>{group}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Annotation Status</InputLabel>
                <Select
                  value={filters.status_filter}
                  onChange={(e) => onFilterChange('status_filter', e.target.value)}
                  label="Annotation Status"
                  sx={{
                    '& .MuiSelect-select': {
                      whiteSpace: 'nowrap',
                      overflow: 'visible',
                      textOverflow: 'unset',
                    },
                  }}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 300,
                        width: 'auto',
                        minWidth: 200,
                      },
                    },
                  }}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="annotated">Annotated</MenuItem>
                  <MenuItem value="not_annotated">Not Annotated</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Training Status</InputLabel>
                <Select
                  value={filters.training_filter}
                  onChange={(e) => onFilterChange('training_filter', e.target.value)}
                  label="Training Status"
                  sx={{
                    '& .MuiSelect-select': {
                      whiteSpace: 'nowrap',
                      overflow: 'visible',
                      textOverflow: 'unset',
                    },
                  }}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 300,
                        width: 'auto',
                        minWidth: 200,
                      },
                    },
                  }}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="trained">Trained</MenuItem>
                  <MenuItem value="not_trained">Not Trained</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Date From"
                type="date"
                value={filters.date_from}
                onChange={(e) => onFilterChange('date_from', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Date To"
                type="date"
                value={filters.date_to}
                onChange={(e) => onFilterChange('date_to', e.target.value)}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  startIcon={<Clear />}
                  onClick={onClearFilters}
                  variant="outlined"
                  sx={{ textTransform: "none" }}
                >
                  Clear Filters
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </FilterCard>
    </Collapse>
  );
}