// pages/Home.js
import React from 'react';
import { Box, Grid, Card } from '@mui/material';
const CaptureImage = () => {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        <Grid size={{ xs: 12, md: 9 }}>
            <Card variant="outlined" sx={{ width: '100%' }}>
            </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 3 }}>
            <Card variant="outlined" sx={{ width: '100%' }}>
            </Card>
        </Grid>
    </Grid>
    </Box>
  );
};

export default CaptureImage;
