// src/pages/auth/SignUp.js
import * as React from 'react';
import AppTheme from '../../../shared-theme/AppTheme';
import SignUpLayout from '../../../components/auth/signup/SignUpLayout';
import SignUpCard from '../../../components/auth/signup/SignUpCard';

export default function SignUp(props) {
  return (
    <AppTheme {...props}>
      <SignUpLayout>
        <SignUpCard />
      </SignUpLayout>
    </AppTheme>
  );
}
