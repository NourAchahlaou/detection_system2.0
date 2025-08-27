import React from 'react';
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Button,
  Typography,
} from "@mui/material";

export default function ConfirmationDialog({
  open,
  onClose, // This function should receive a boolean (confirm/cancel)
  selectedCount,
  actionType, // Add this prop to handle different action types
  targetName, // Add target name for better messaging
}) {
  
  // Get the appropriate title and message based on action type
  const getDialogContent = () => {
    switch (actionType) {
      case 'delete':
        return {
          title: 'Delete Piece',
          message: `Are you sure you want to delete "${targetName || 'this piece'}"? This action cannot be undone.`
        };
      case 'bulkDelete':
        return {
          title: 'Delete Selected Items',
          message: `Are you sure you want to delete ${selectedCount} selected item${selectedCount !== 1 ? 's' : ''}? This action cannot be undone.`
        };
      default:
        return {
          title: 'Confirm Action',
          message: 'Are you sure you want to proceed? This action cannot be undone.'
        };
    }
  };

  const { title, message } = getDialogContent();

  return (
    <Dialog
      open={open}
      onClose={() => onClose(false)} // Cancel action
      PaperProps={{
        sx: { borderRadius: "16px", padding: "8px" }
      }}
    >
      <DialogTitle sx={{ fontWeight: "600", color: "#333" }}>
        {title}
      </DialogTitle>
      <DialogContent>
        <Typography>
          {message}
        </Typography>
      </DialogContent>
      <DialogActions sx={{ gap: 1, padding: "16px 24px" }}>
        <Button 
          onClick={() => onClose(false)} // Cancel
          sx={{ textTransform: "none" }}
        >
          Cancel
        </Button>
        <Button 
          onClick={() => onClose(true)} // Confirm
          color="error"
          variant="contained"
          sx={{ textTransform: "none" }}
        >
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
}