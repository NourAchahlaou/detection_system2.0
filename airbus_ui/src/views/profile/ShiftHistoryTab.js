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
  Chip
} from '@mui/material';

export default function ShiftHistoryTab() {
  const shiftData = [
    { date: "07/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 278, status: "Completed" },
    { date: "06/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 295, status: "Completed" },
    { date: "05/05/2025", start: "07:45", end: "15:30", hours: "07h 45min", station: "Station 2", inspected: 265, status: "Completed" },
    { date: "04/05/2025", start: "08:15", end: "16:30", hours: "08h 15min", station: "Station 1", inspected: 302, status: "Completed" },
    { date: "03/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 287, status: "Completed" },
    { date: "02/05/2025", start: "07:30", end: "15:45", hours: "08h 15min", station: "Station 2", inspected: 310, status: "Completed" },
    { date: "01/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 292, status: "Completed" },
  ];

  return (
    <Box>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
        <Typography variant="subtitle1" fontWeight="medium">Shift History</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              defaultValue="may"
              size="small"
            >
              <MenuItem value="may">May 2025</MenuItem>
              <MenuItem value="april">April 2025</MenuItem>
              <MenuItem value="march">March 2025</MenuItem>
            </Select>
          </FormControl>
          <Button variant="outlined" size="small" color="primary">Export</Button>
        </Box>
      </Box>
      
      <TableContainer>
        <Table sx={{ minWidth: 650 }} size="small">
          <TableHead>
            <TableRow>
              <TableCell>Date</TableCell>
              <TableCell>Shift Start</TableCell>
              <TableCell>Shift End</TableCell>
              <TableCell>Total Hours</TableCell>
              <TableCell>Station</TableCell>
              <TableCell>Inspected</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {shiftData.map((shift, index) => (
              <TableRow key={index} hover>
                <TableCell>{shift.date}</TableCell>
                <TableCell>{shift.start}</TableCell>
                <TableCell>{shift.end}</TableCell>
                <TableCell>{shift.hours}</TableCell>
                <TableCell>{shift.station}</TableCell>
                <TableCell>{shift.inspected}</TableCell>
                <TableCell>
                  <Chip
                    label={shift.status}
                    color="success"
                    size="small"
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        component="div"
        count={12}
        page={0}
        onPageChange={() => {}}
        rowsPerPage={7}
        onRowsPerPageChange={() => {}}
        rowsPerPageOptions={[7, 14, 21]}
      />
    </Box>
  );
}