import React from 'react';
import {
  TableRow,
  TableBody,
  TableCell,
  TableHead,
  Checkbox,
  Box,
  Avatar,
  Typography,
  Chip,
  Button,
  TablePagination,
} from "@mui/material";
import {
  Delete, Visibility, PhotoLibrary, CheckCircle, RadioButtonUnchecked,
  Image as ImageIcon,
} from "@mui/icons-material";
import { ProductTable, ModernCard, StatusChip, ActionButton } from './StyledComponents';

export default function DataTable({ 
  datasets, 
  selectedDatasets, 
  selectAll, 
  onSelectAll, 
  onSelect, 
  onView, 
  onDelete, 
  onTrain,
  trainingInProgress,
  page,
  pageSize,
  totalCount,
  onPageChange,
  onRowsPerPageChange,
  formatDate
}) {
  return (
    <ModernCard elevation={0}>
      <ProductTable>
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox">
              <Checkbox
                checked={selectAll}
                onChange={onSelectAll}
                sx={{ color: "#667eea", '&.Mui-checked': { color: "#667eea" } }}
              />
            </TableCell>
            <TableCell>Piece Details</TableCell>
            <TableCell align="center">Group</TableCell>
            <TableCell align="center">Images</TableCell>
            <TableCell align="center">Annotations</TableCell>
            <TableCell align="center">Status</TableCell>
            <TableCell align="center">Training</TableCell>
            <TableCell align="center">Created</TableCell>
            <TableCell align="center">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {datasets.map((piece) => (
            <TableRow key={piece.id} hover>
              <TableCell padding="checkbox">
                <Checkbox
                  checked={selectedDatasets.includes(piece.id)}
                  onChange={() => onSelect(piece.id)}
                  sx={{ color: "#667eea", '&.Mui-checked': { color: "#667eea" } }}
                />
              </TableCell>
              
              <TableCell>
                <Box display="flex" alignItems="center">
                  <Avatar 
                    sx={{ 
                      width: 40, 
                      height: 40, 
                      mr: 2, 
                      borderRadius: "8px",
                      bgcolor: "rgba(102, 126, 234, 0.1)",
                      color: "#667eea"
                    }}
                  >
                    <PhotoLibrary />
                  </Avatar>
                  <Box>
                    <Typography variant="body2" fontWeight="600" color="#333">
                      {piece.label}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      ID: {piece.class_data_id}
                    </Typography>
                  </Box>
                </Box>
              </TableCell>
              
              <TableCell align="center">
                <Chip 
                  label={piece.group} 
                  size="small"
                  sx={{ 
                    bgcolor: "rgba(102, 126, 234, 0.1)",
                    color: "#667eea",
                    fontWeight: "600"
                  }}
                />
              </TableCell>
              
              <TableCell align="center">
                <Box display="flex" alignItems="center" justifyContent="center" gap={0.5}>
                  <ImageIcon fontSize="small" color="action" />
                  <Typography variant="body2" fontWeight="600">
                    {piece.nbre_img}
                  </Typography>
                </Box>
              </TableCell>
              
              <TableCell align="center">
                <Typography variant="body2" color="#667eea" fontWeight="600">
                  {piece.total_annotations}
                </Typography>
              </TableCell>
              
              <TableCell align="center">
                <StatusChip 
                  variant={piece.is_annotated ? "completed" : "pending"}
                  icon={piece.is_annotated ? <CheckCircle /> : <RadioButtonUnchecked />}
                  label={piece.is_annotated ? "Annotated" : "Pending"}
                  size="small"
                />
              </TableCell>
              
              <TableCell align="center">
                <StatusChip 
                  variant={piece.is_yolo_trained ? "trained" : "pending"}
                  icon={piece.is_yolo_trained ? <CheckCircle /> : <RadioButtonUnchecked />}
                  label={piece.is_yolo_trained ? "Trained" : "Not Trained"}
                  size="small"
                />
              </TableCell>
              
              <TableCell align="center">
                <Typography variant="caption" color="text.secondary">
                  {formatDate(piece.created_at)}
                </Typography>
              </TableCell>
              
              <TableCell align="center">
                <Box display="flex" justifyContent="center" gap={0.5}>
                  <ActionButton variant="view" onClick={() => onView(piece)}>
                    <Visibility fontSize="small" />
                  </ActionButton>
                  <ActionButton variant="delete" onClick={() => onDelete(piece)}>
                    <Delete fontSize="small" />
                  </ActionButton>
                  <Button 
                    onClick={() => onTrain(piece)} 
                    size="small"
                    variant="outlined"
                    disabled={trainingInProgress}
                    sx={{ 
                      textTransform: "none",
                      minWidth: "60px",
                      fontSize: "0.75rem",
                      py: 0.5
                    }}
                  >
                    {trainingInProgress ? "Training..." : "Train"}
                  </Button>
                </Box>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </ProductTable>
      
      <TablePagination
        component="div"
        count={totalCount}
        page={page}
        onPageChange={onPageChange}
        rowsPerPage={pageSize}
        onRowsPerPageChange={onRowsPerPageChange}
        rowsPerPageOptions={[5, 10, 25, 50]}
        sx={{
          borderTop: "1px solid rgba(102, 126, 234, 0.1)",
          "& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows": {
            color: "#667eea",
            fontWeight: "500"
          }
        }}
      />
    </ModernCard>
  );
}