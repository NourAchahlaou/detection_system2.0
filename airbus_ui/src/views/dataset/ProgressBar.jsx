import React from 'react';
import { Modal, Box, Typography, LinearProgress, Button, IconButton } from '@mui/material';
import { Close } from '@mui/icons-material';
import axios from 'axios'; // Import axios for making HTTP requests

const modalStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

export default function TrainingProgressModal({ open, handleClose, progress, details }) {
  const handleStopTraining = async () => {
    try {
      await axios.post('http://localhost:8000/detection/stop_training');
      // Handle any state updates or notifications after stopping the training
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      aria-labelledby="training-progress-modal"
      aria-describedby="training-progress-modal-description"
    >
      <Box sx={modalStyle}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" id="training-progress-modal">
            Training Progress
          </Typography>
          <IconButton onClick={handleClose}>
            <Close />
          </IconButton>
        </Box>
        <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {details}
        </Typography>
        <Button variant="contained" color="error" onClick={handleStopTraining}>
          Stop Training
        </Button>
      </Box>
    </Modal>
  );
}
