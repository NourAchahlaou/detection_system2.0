// src/pages/auth/SignInSide.js
import * as React from 'react';
import AppTheme from '../../../shared-theme/AppTheme';
import SignInLayout from '../../../components/auth/signin/layout';
import SignInCard from '../../../components/auth/signin/SignInCard';
import Content from '../../../components/auth/signin/Content';

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
