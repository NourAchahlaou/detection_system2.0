import * as React from 'react';
import { 
  Box, 
  Card,
  CardContent,
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
  
} from '@mui/material';

export default function ShiftHistoryTab() {
  const shiftData = [
    { date: "07/05/2025", start: "08:00", end: "16:00", hours: "08h 00min", station: "Station 3", inspected: 278},

  ];

  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
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
          rowsPerPage={10}
          onRowsPerPageChange={() => {}}
          rowsPerPageOptions={[7, 14, 21]}
        />
      </CardContent>
    </Card>
  );
}