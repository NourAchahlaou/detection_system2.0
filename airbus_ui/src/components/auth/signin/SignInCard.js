import * as React from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import MuiCard from '@mui/material/Card';
import Checkbox from '@mui/material/Checkbox';
import Divider from '@mui/material/Divider';
import FormLabel from '@mui/material/FormLabel';
import FormControl from '@mui/material/FormControl';
import FormControlLabel from '@mui/material/FormControlLabel';
import Link from '@mui/material/Link';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { styled } from '@mui/material/styles';
import ForgotPassword from './ForgotPassword';
import { GoogleIcon } from '../CustomIcons';
import { ReactComponent as AirVisionLogo } from '../../../assets/Airvisionlogo_updated.svg';

const Card = styled(MuiCard)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'center',
  width: '100%',
  padding: theme.spacing(4),
  gap: theme.spacing(2),
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

export default function SignInCard() {
  const navigate = useNavigate();
  const [emailError, setEmailError] = React.useState(false);
  const [emailErrorMessage, setEmailErrorMessage] = React.useState('');
  const [passwordError, setPasswordError] = React.useState(false);
  const [passwordErrorMessage, setPasswordErrorMessage] = React.useState('');
  const [open, setOpen] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [snackbar, setSnackbar] = React.useState({
    open: false,
    message: '',
    severity: 'info'
  });

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };
  
  
  const validateInputs = () => {
    const email = document.getElementById('email');
    const password = document.getElementById('password');

    let isValid = true;

    if (!email.value || !/\S+@\S+\.\S+/.test(email.value)) {
      setEmailError(true);
      setEmailErrorMessage('Please enter a valid email address.');
      isValid = false;
    } else {
      setEmailError(false);
      setEmailErrorMessage('');
    }

    if (!password.value) {
      setPasswordError(true);
      setPasswordErrorMessage('Password is required.');
      isValid = false;
    } else {
      setPasswordError(false);
      setPasswordErrorMessage('');
    }
  
    return isValid;
  };

  // New function to check profile completion status
  const checkProfileCompletion = async (accessToken) => {
    try {
      const response = await fetch('http://localhost:8001/auth/completion', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const completionData = await response.json();
        console.log('Profile completion data:', completionData);
        
        // Check if profile is complete (100%)
        if (completionData.completion_percentage === 100) {
          // Profile is complete, navigate to dashboard
          navigate('/dashboard');
        } else {
          // Profile is incomplete, navigate to profile completion page
          setSnackbar({
            open: true,
            message: 'Please complete your profile before continuing.',
            severity: 'info'
          });
          
          // Navigate to profile page with missing fields info
          navigate('/auth/profile', { 
            state: { 
              missingFields: completionData.missing_fields,
              completionPercentage: completionData.completion_percentage
            } 
          });
        }
      } else {
        // If there's an error fetching completion data, redirect to profile page by default
        console.error('Error fetching profile completion data');
        navigate('/auth/profile');
      }
    } catch (error) {
      console.error('Error checking profile completion:', error);
      navigate('/auth/profile');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    // Validate form inputs
    if (!validateInputs()) {
      return;
    }

    setIsLoading(true);
    const formData = new FormData(event.currentTarget);
    
    try {
      // Create FormData object for FastAPI's Form dependencies
      const formDataForApi = new URLSearchParams();
      formDataForApi.append('email', formData.get('email'));
      formDataForApi.append('password', formData.get('password'));
      
      const response = await fetch('http://localhost:8001/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formDataForApi,
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Login successful:', result);
        
        // Store tokens in localStorage
        localStorage.setItem('accessToken', result.access_token);
        localStorage.setItem('refreshToken', result.refresh_token);
        
        setSnackbar({
          open: true,
          message: 'Login successful!',
          severity: 'success'
        });
        
        // Check profile completion before redirecting
        await checkProfileCompletion(result.access_token);
        
      } else {
        const errorResult = await response.json();
        console.error('Login failed:', errorResult);
        
        // Check if account is not verified
        if (errorResult.detail && errorResult.detail.includes('not verified')) {
          setSnackbar({
            open: true,
            message: 'Your account is not verified. Please check your email for verification instructions.',
            severity: 'warning'
          });
        } else {
          setSnackbar({
            open: true,
            message: `Login failed: ${errorResult.detail || 'Invalid credentials'}`,
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
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card variant="outlined">
      <Box sx={{ display: { xs: 'none', md: 'flex' }, color: '#00205B', marginBottom: -9, marginTop: -10, flexDirection: 'column', alignSelf: 'center' }}>
        <AirVisionLogo width={150} height={200} />
      </Box>
      <Typography
        component="h1"
        variant="h4"
        sx={{ width: '100%', fontSize: 'clamp(2rem, 10vw, 2.15rem)' }}
      >
        Sign in
      </Typography>
      <Box
        component="form"
        onSubmit={handleSubmit}
        noValidate
        sx={{ display: 'flex', flexDirection: 'column', width: '100%', gap: 2 }}
      >
        <FormControl>
          <FormLabel htmlFor="email">Email</FormLabel>
          <TextField
            error={emailError}
            helperText={emailErrorMessage}
            id="email"
            type="email"
            name="email"
            placeholder="your@email.com"
            autoComplete="email"
            autoFocus
            required
            fullWidth
            variant="outlined"
            color={emailError ? 'error' : 'transparent'}
          />
        </FormControl>
        <FormControl>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <FormLabel htmlFor="password">Password</FormLabel>
            <Link
              component="button"
              type="button"
              onClick={handleClickOpen}
              variant="body2"
              sx={{ alignSelf: 'baseline' }}
            >
              Forgot your password?
            </Link>
          </Box>
          <TextField
            error={passwordError}
            helperText={passwordErrorMessage}
            name="password"
            placeholder="••••••"
            type="password"
            id="password"
            autoComplete="current-password"
            required
            fullWidth
            variant="outlined"
            color={passwordError ? 'error' : 'transparent'}
          />
        </FormControl>
        <FormControlLabel
          control={<Checkbox value="remember" color="transparent" />}
          label="Remember me"
        />
        <ForgotPassword open={open} handleClose={handleClose} />
        <Button 
          type="submit" 
          fullWidth 
          variant="contained"
          disabled={isLoading}
        >
          {isLoading ? 'Please wait...' : 'Sign in'}
        </Button>
        <Typography sx={{ textAlign: 'center' }}>
          Don&apos;t have an account?{' '}
          <Link href="/auth/signup" variant="body2">
            Sign up
          </Link>
        </Typography>
      </Box>
      <Divider>or</Divider>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Button
          fullWidth
          variant="outlined"
          onClick={() => alert('Sign in with Google')}
          startIcon={<GoogleIcon />}
          disabled={isLoading}
        >
          Sign in with Google
        </Button>
      </Box>
      
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
    </Card>
  );
}