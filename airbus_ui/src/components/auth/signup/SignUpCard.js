import * as React from 'react';
import { useNavigate } from 'react-router-dom';

import {
  Box,
  Button,
  Checkbox,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  Link,
  TextField,
  Typography,
  Card as MuiCard,
  Snackbar,
  Alert,
  Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { GoogleIcon } from '../CustomIcons';
import { ReactComponent as AirVisionLogo } from '../../../assets/Airvisionlogo_updated.svg';
import VerificationCodeDialog from './VerificationCodeDialog';

const Card = styled(MuiCard)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'center',
  width: '100%',
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  margin: 'auto',
  boxShadow:
    'hsla(220, 30%, 5%, 0.05) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.05) 0px 15px 35px -5px',
  [theme.breakpoints.up('sm')]: {
    width: '450px',
  },
  ...theme.applyStyles('dark', {
    boxShadow:
      'hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px',
  }),
}));

// User Exists Card component
const UserExistsCard = ({ email, onClose, onNavigateToLogin }) => {
  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        maxWidth: '450px',
        mx: 'auto',
        mt: 2
      }}
    >
      <Typography variant="h5" component="h2">
        Account Already Exists
      </Typography>
      <Typography variant="body1">
        An account with email <strong>{email}</strong> already exists. Would you like to sign in instead?
      </Typography>
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        <Button variant="outlined" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="contained" onClick={onNavigateToLogin}>
          Go to Sign In
        </Button>
      </Box>
    </Paper>
  );
};

export default function SignUpCard() {
  const navigate = useNavigate();
  const [emailError, setEmailError] = React.useState(false);
  const [emailErrorMessage, setEmailErrorMessage] = React.useState('');
  const [passwordError, setPasswordError] = React.useState(false);
  const [passwordErrorMessage, setPasswordErrorMessage] = React.useState('');
  const [nameError, setNameError] = React.useState(false);
  const [nameErrorMessage, setNameErrorMessage] = React.useState('');
  const [verificationDialogOpen, setVerificationDialogOpen] = React.useState(false);
  const [currentEmail, setCurrentEmail] = React.useState('');
  const [userExistsModalOpen, setUserExistsModalOpen] = React.useState(false);
  const [snackbar, setSnackbar] = React.useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  const SPECIAL_CHARACTERS = /[!@#$%^&*(),.?":{}|<>]/;

  const validateInputs = () => {
    const email = document.getElementById('email');
    const password = document.getElementById('password');
    const name = document.getElementById('name');

    let isValid = true;

    if (!email.value || !/\S+@\S+\.\S+/.test(email.value)) {
      setEmailError(true);
      setEmailErrorMessage('Please enter a valid email address.');
      isValid = false;
    } else {
      setEmailError(false);
      setEmailErrorMessage('');
    }

    if (password.value.length < 8) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must be at least 8 characters long.');
      isValid = false;
    } else if (!/[A-Z]/.test(password.value)) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must contain at least one uppercase letter.');
      isValid = false;
    } else if (!/[a-z]/.test(password.value)) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must contain at least one lowercase letter.');
      isValid = false;
    } else if (!/\d/.test(password.value)) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must contain at least one number.');
      isValid = false;
    } else if (!SPECIAL_CHARACTERS.test(password.value)) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must contain at least one special character.');
      isValid = false;
    } else {
      setPasswordError(false);
      setPasswordErrorMessage('');
    }

    if (!name.value || name.value.length < 1) {
      setNameError(true);
      setNameErrorMessage('Name is required.');
      isValid = false;
    } else {
      setNameError(false);
      setNameErrorMessage('');
    }

    return isValid;
  };

// Modified loginUser function to use FormData instead of JSON
const loginUser = async (credentials) => {
  try {
    console.log('Attempting login with:', { email: credentials.email, passwordLength: credentials.password?.length || 0 });
    
    // Create FormData object - this is critical for working with FastAPI's Form dependencies
    const formData = new URLSearchParams();
    formData.append('email', credentials.email);
    formData.append('password', credentials.password);
    
    const loginResponse = await fetch('http://localhost:8001/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    if (loginResponse.ok) {
      const loginResult = await loginResponse.json();
      console.log('Login successful:', loginResult);
      
      // Store tokens in localStorage
      localStorage.setItem('accessToken', loginResult.access_token);
      localStorage.setItem('refreshToken', loginResult.refresh_token);
      
      // Clear temporary stored password
      localStorage.removeItem('tempPassword');
      
      // Show success message
      setSnackbar({
        open: true,
        message: 'Account verified and logged in successfully!',
        severity: 'success'
      });
      
      // Redirect to profile completion
      setTimeout(() => {
        navigate('/auth/profile');
      }, 1500);
      
      return loginResult;
    } else {
      // Try to get response details for better error handling
      let errorData;
      try {
        errorData = await loginResponse.json();
      } catch (e) {
        errorData = { detail: `HTTP ${loginResponse.status}` };
      }
      
      console.error('Login failed:', errorData);
      
      // Check if the account is not verified
      if (errorData.detail && errorData.detail.includes('not verified')) {
        // Store current email for verification dialog
        setCurrentEmail(credentials.email);
        // Store password temporarily for after verification
        localStorage.setItem('tempPassword', credentials.password);
        // Open verification dialog
        setVerificationDialogOpen(true);
        return null;
      }
      
      throw new Error(errorData.detail || 'Login failed');
    }
  } catch (error) {
    console.error('Login error:', error);
    setSnackbar({
      open: true,
      message: error.message || 'An error occurred during login',
      severity: 'error'
    });
    throw error;
  }
};
// Updated handleVerifyAccount function to properly handle login after verification
const handleVerifyAccount = async (email, code) => {
  try {
    const response = await fetch('http://localhost:8001/users/verify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, token: code }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log('Account verified successfully:', result);
      
      // Get the saved password
      const savedPassword = localStorage.getItem('tempPassword');
      
      if (!savedPassword) {
        throw new Error('Could not retrieve password for automatic login');
      }
      
      // Now try to login with proper credentials
      return await loginUser({ 
        email: email, 
        password: savedPassword 
      });
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Verification failed');
    }
  } catch (error) {
    console.error('Error verifying account:', error);
    setSnackbar({
      open: true,
      message: error.message || 'Verification failed',
      severity: 'error'
    });
    throw error;
  }
};

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Validate form inputs
    if (!validateInputs()) {
      return;
    }

    const formData = new FormData(event.currentTarget);
    const data = {
      name: formData.get('name'),
      email: formData.get('email'),
      password: formData.get('password'),
    };

    // Store email for potential user exists modal
    setCurrentEmail(data.email);

    // Make API request to register user
    try {
      const response = await fetch('http://localhost:8001/users/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('User registered successfully:', result);
        
        setSnackbar({
          open: true,
          message: 'Registration successful! Please verify your account.',
          severity: 'success'
        });
        
        // Store password temporarily for after verification
        localStorage.setItem('tempPassword', data.password);
        
        // Open verification dialog directly after successful registration
        setVerificationDialogOpen(true);
      } else {
        const errorResult = await response.json();
        console.error('Error registering user:', errorResult);
        
        // Check if error is about email already existing
        if (errorResult.detail && errorResult.detail.includes("Email is already exists")) {
          // Open user exists modal instead of showing error snackbar
          setUserExistsModalOpen(true);
        } else {
          setSnackbar({
            open: true,
            message: `Registration failed: ${errorResult.detail || 'Unknown error'}`,
            severity: 'error'
          });
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setSnackbar({
        open: true,
        message: 'An error occurred. Please try again later.',
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleNavigateToLogin = () => {
    navigate('/auth/login');
  };

  const handleCloseUserExistsModal = () => {
    setUserExistsModalOpen(false);
  };

  return (
    <>
      {userExistsModalOpen ? (
        <UserExistsCard 
          email={currentEmail}
          onClose={handleCloseUserExistsModal}
          onNavigateToLogin={handleNavigateToLogin}
        />
      ) : (
        <Card variant="outlined">
          <Box sx={{ display: { xs: 'none', md: 'flex' }, color: '#00205B', marginBottom: -9, marginTop: -10, flexDirection: 'column', alignSelf: 'center' }}>
            <AirVisionLogo width={150} height={200} />
          </Box>
          <Typography component="h1" variant="h4">
            Sign up
          </Typography>
          <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <FormControl>
              <FormLabel htmlFor="name">Full name</FormLabel>
              <TextField
                autoComplete="name"
                name="name"
                required
                fullWidth
                id="name"
                placeholder="Jon Snow"
                error={nameError}
                helperText={nameErrorMessage}
                color={nameError ? 'error' : 'transparent'}
              />
            </FormControl>
            <FormControl>
              <FormLabel htmlFor="email">Email</FormLabel>
              <TextField
                required
                fullWidth
                id="email"
                placeholder="your@email.com"
                name="email"
                autoComplete="email"
                error={emailError}
                helperText={emailErrorMessage}
                color={emailError ? 'error' : 'transparent'}
              />
            </FormControl>
            <FormControl>
              <FormLabel htmlFor="password">Password</FormLabel>
              <TextField
                required
                fullWidth
                name="password"
                placeholder="••••••"
                type="password"
                id="password"
                autoComplete="new-password"
                error={passwordError}
                helperText={passwordErrorMessage}
                color={passwordError ? 'error' : 'transparent'}
              />
            </FormControl>
            <FormControlLabel
              control={<Checkbox value="allowExtraEmails" color="primary" />}
              label="I want to receive updates via email."
            />
            <Button type="submit" fullWidth variant="contained">
              Sign up
            </Button>
          </Box>
          <Divider>
            <Typography sx={{ color: 'text.secondary' }}>or</Typography>
          </Divider>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Button
              fullWidth
              variant="outlined"
              onClick={() => alert('Sign up with Google')}
              startIcon={<GoogleIcon />}
            >
              Sign up with Google
            </Button>

            <Typography sx={{ textAlign: 'center' }}>
              Already have an account?{' '}
              <Link href="/auth/login" variant="body2">
                Sign in
              </Link>
            </Typography>
          </Box>
        </Card>
      )}
      
      {/* Verification Code Dialog */}
      <VerificationCodeDialog
        open={verificationDialogOpen}
        onClose={() => setVerificationDialogOpen(false)}
        onVerify={handleVerifyAccount}
        email={currentEmail}
      />
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
}