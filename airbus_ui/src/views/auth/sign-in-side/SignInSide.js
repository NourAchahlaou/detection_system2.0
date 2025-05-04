// src/pages/auth/SignInSide.js
import * as React from 'react';
import AppTheme from '../../../shared-theme/AppTheme';
import SignInLayout from '../../../components/auth/layout';
import SignInCard from '../../../components/auth/SignInCard';
import Content from '../../../components/auth/Content';

export default function SignInSide(props) {
  return (
    <AppTheme {...props}>
      <SignInLayout>
        <Content />
        <SignInCard />
      </SignInLayout>
    </AppTheme>
  );
}
