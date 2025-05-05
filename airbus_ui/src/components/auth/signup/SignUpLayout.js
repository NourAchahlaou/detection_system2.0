// src/components/auth/signup/SignUpLayout.js
import { Stack, styled } from '@mui/material';

const SignUpContainer = styled(Stack)(({ theme }) => ({
  height: '100vh', // Fill full viewport height
  minHeight: '100vh',
  padding: theme.spacing(2),
  alignItems: 'center', // Center horizontally
  justifyContent: 'center', // Center vertically
  position: 'relative',
  overflow: 'hidden', // Prevent scrolling if not needed
  [theme.breakpoints.up('sm')]: {
    padding: theme.spacing(4),
  },
  '&::before': {
    content: '""',
    display: 'block',
    position: 'absolute',
    zIndex: -1,
    inset: 0,
    backgroundImage:
      'radial-gradient(ellipse at 50% 50%, hsl(210, 100%, 97%), hsl(0, 0%, 100%))',
    backgroundRepeat: 'no-repeat',
    ...theme.applyStyles('dark', {
      backgroundImage:
        'radial-gradient(at 50% 50%, hsla(210, 100%, 16%, 0.5), hsl(220, 30%, 5%))',
    }),
  },
}));

export default function SignUpLayout({ children }) {
  return (
    <SignUpContainer>
      {children}
    </SignUpContainer>
  );
}

