import * as React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  FormControl, 
  Select, 
  MenuItem, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  TablePagination,
  Chip,
  Card,
  CardContent,
  IconButton
} from '@mui/material';
import { 
  Visibility as EyeIcon, 
  FindInPage as FileSearchIcon 
} from '@mui/icons-material';

export default function ActivitiesTab() {
  const activityData = [
    { time: "14:23", action: "Inspected", pieceRef: "D532.31953.010.10", lot: "L7841", camera: "Top", detection: "OK", confidence: "98%" },
    { time: "14:00", action: "Flagged", pieceRef: "D532.31953.012.10", lot: "L7842", camera: "Side", detection: "Wrong Lot", confidence: "95%" },

  ];

  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="subtitle1" fontWeight="medium">Activity Log</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <Select
              defaultValue="all"
              size="small"
            >
              <MenuItem value="all">All Activities</MenuItem>
              <MenuItem value="inspected">Inspected</MenuItem>
              <MenuItem value="flagged">Flagged</MenuItem>
              <MenuItem value="verified">Verified</MenuItem>
            </Select>
          </FormControl>
          <Button variant="outlined" size="small" color="primary">Filter</Button>
        </Box>
      </Box>
      
      <TableContainer>
        <Table sx={{ minWidth: 650 }} size="small">
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>Piece Ref</TableCell>
              <TableCell>Lot</TableCell>
              <TableCell>Camera</TableCell>
              <TableCell>Detection</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {activityData.map((activity, index) => (
              <TableRow key={index} hover>
                <TableCell>{activity.time}</TableCell>
                <TableCell>
                  <Chip
                    label={activity.action}
                    color={
                      activity.action === "Inspected" ? "primary" : 
                      activity.action === "Verified" ? "success" : 
                      "error"
                    }
                    size="small"
                  />
                </TableCell>
                <TableCell>{activity.pieceRef}</TableCell>
                <TableCell>{activity.lot}</TableCell>
                <TableCell>{activity.camera}</TableCell>
                <TableCell>
                  <Chip
                    label={`${activity.detection} (${activity.confidence})`}
                    color={
                      activity.detection === "OK" ? "success" : 
                      activity.detection === "Wrong Lot" ? "warning" : 
                      "error"
                    }
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <IconButton size="small" color="primary">
                    <EyeIcon fontSize="small" />
                  </IconButton>
                  <IconButton size="small" color="default">
                    <FileSearchIcon fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        component="div"
        count={278}
        page={0}
        onPageChange={() => {}}
        rowsPerPage={7}
        onRowsPerPageChange={() => {}}
        rowsPerPageOptions={[7, 14, 21]}
      />
    </CardContent>
    </Card>
  );
}