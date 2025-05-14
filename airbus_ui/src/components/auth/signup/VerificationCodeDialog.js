import * as React from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  CircularProgress
} from '@mui/material';
import api from '../../../utils/UseAxios'; 

export default function VerificationCodeDialog({ open, onClose, onVerify, email }) {
  const [code, setCode] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  const handleSubmit = async () => {
    if (!code.trim()) {
      setError('Please enter the verification code');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      await onVerify(email, code);
      onClose();
    } catch (error) {
      setError(error.message || 'Verification failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    setIsLoading(true);
    setError('');

    try {
      const response = await api.post('/api/users/users/resend-verification',JSON.stringify({ email }), {

        headers: {
          'Content-Type': 'application/json',
        },
  
      });

      if (response.ok) {
        setError('Verification code resent successfully!');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to resend verification code');
      }
    } catch (error) {
      setError(error.message || 'An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Reset state when dialog opens
  React.useEffect(() => {
    if (open) {
      setCode('');
      setError('');
    }
  }, [open]);

  return (
    <Dialog open={open} onClose={onClose} aria-labelledby="verification-dialog-title">
      <DialogTitle id="verification-dialog-title">Verify Your Account</DialogTitle>
      <DialogContent>
        <DialogContentText>
          We sent a verification code to <strong>{email}</strong>. Please enter the code below to verify your account.
        </DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          id="verification-code"
          label="Verification Code"
          type="text"
          fullWidth
          variant="outlined"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          error={!!error}
          helperText={error}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={handleResendCode} disabled={isLoading}>
          Resend Code
        </Button>
        <Button onClick={onClose} disabled={isLoading}>
          Cancel
        </Button>
        <Button 
          onClick={handleSubmit} 
          variant="contained" 
          disabled={isLoading || !code.trim()}
          startIcon={isLoading ? <CircularProgress size={20} /> : null}
        >
          {isLoading ? 'Verifying...' : 'Verify'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}