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
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { GoogleIcon } from '../CustomIcons';
import { ReactComponent as AirVisionLogo } from '../../../assets/Airvisionlogo_updated.svg';

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
export default function SignUpCard() {
  const navigate = useNavigate();
  const [emailError, setEmailError] = React.useState(false);
  const [emailErrorMessage, setEmailErrorMessage] = React.useState('');
  const [passwordError, setPasswordError] = React.useState(false);
  const [passwordErrorMessage, setPasswordErrorMessage] = React.useState('');
  const [nameError, setNameError] = React.useState(false);
  const [nameErrorMessage, setNameErrorMessage] = React.useState('');
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

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Validate form inputs
    if (nameError || emailError || passwordError) {
      return;
    }


    const formData = new FormData(event.currentTarget);
    const data = {
      name: formData.get('name'),
      email: formData.get('email'),
      password: formData.get('password'),
    };

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
              // Redirect to profile completion
        navigate('/complete-profile');
        // Handle successful registration (e.g., redirect to login page)
      } else {
        const errorResult = await response.json();
        console.error('Error registering user:', errorResult);
        // Handle error (e.g., display error message)
      }
    } catch (error) {
      console.error('Error:', error);
      // Handle network or other errors
    }
  };

  return (
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
            color={nameError ? 'error' : 'primary'}
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
            color={emailError ? 'error' : 'primary'}
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
            color={passwordError ? 'error' : 'primary'}
          />
        </FormControl>
        <FormControlLabel
          control={<Checkbox value="allowExtraEmails" color="primary" />}
          label="I want to receive updates via email."
        />
        <Button type="submit" fullWidth variant="contained" onClick={validateInputs}>
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
          <Link href="/material-ui/getting-started/templates/sign-in/" variant="body2">
            Sign in
          </Link>
        </Typography>
      </Box>
    </Card>
  );
}
