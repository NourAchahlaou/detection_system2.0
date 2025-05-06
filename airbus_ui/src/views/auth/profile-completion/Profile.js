import * as React from 'react';
import MyProfile from '../../../components/auth/profileCompletion/MyProfile';
import AppTheme from '../../../shared-theme/AppTheme';
import SignUpLayout from '../../../components/auth/signup/SignUpLayout';

export default function Profile(props) {
  return (
    <AppTheme {...props}>
      <SignUpLayout>
        <MyProfile />
      </SignUpLayout>
    </AppTheme>
  );
}
