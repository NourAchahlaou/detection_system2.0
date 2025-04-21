import * as React from 'react';
import { Outlet } from 'react-router-dom'; // ✅ add this
import { alpha } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';

import AppNavbar from './navbar/AppNavbar';
import Header from './header/Header';
import AppTheme from '../../shared-theme/AppTheme';
import Copyright from './footer/Copyright';
import {
  chartsCustomizations,
  dataGridCustomizations,
  datePickersCustomizations,
  treeViewCustomizations,
} from '../../components/theme/customizations';

const xThemeComponents = {
  ...chartsCustomizations,
  ...dataGridCustomizations,
  ...datePickersCustomizations,
  ...treeViewCustomizations,
};
export default function Layout(props) {
  return (
    <AppTheme {...props} themeComponents={xThemeComponents}>
      <CssBaseline enableColorScheme />

      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh', // ensures it stretches full screen height
        }}
      >
        <Box sx={{ display: 'flex', flex: 1 }}>
          <AppNavbar />

          <Box
            component="main"
            sx={(theme) => ({
              flexGrow: 1,
              backgroundColor: theme.vars
                ? `rgba(${theme.vars.palette.background.defaultChannel} / 1)`
                : alpha(theme.palette.background.default, 1),
              overflow: 'auto',
            })}
          >
            <Stack
              spacing={2}
              sx={{
                alignItems: 'center',
                mx: 3,
                pb: 5,
                mt: { xs: 8, md: 0 },
              }}
            >
              <Header />
              <Outlet />
            </Stack>
          </Box>
        </Box>

        {/* ✅ Footer stays at the bottom */}
        <Box component="footer" sx={{ py: 2, textAlign: 'center' }}>
          <Copyright />
        </Box>
      </Box>
    </AppTheme>
  );
}
