import * as React from 'react';
import PropTypes from 'prop-types';
import Avatar from '@mui/material/Avatar';
import Box from '@mui/material/Box';
import OptionsMenu from './OptionsMenu';

import Divider from '@mui/material/Divider';
import Drawer, { drawerClasses } from '@mui/material/Drawer';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import MenuContent from './MenuContent';

function SideMenuMobile({ open, toggleDrawer }) {
  return (
    <Drawer
      anchor="left"
      open={open}
      onClose={toggleDrawer(false)}
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        [`& .${drawerClasses.paper}`]: {
          backgroundImage: 'none',
          
          width : 240,
        },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          mt: 'calc(var(--template-frame-height, 0px) + 4px)',
          p: 1.5,
          gap: '0.75rem', // 👉 adjust spacing here (e.g., 0.5rem, 1rem...)
        }}
      >
        <CustomIcon />
        <Typography
          variant="h4"
          component="h1"
          sx={{
            fontWeight: 600,
            color: 'text.primary',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}
        >
          AIRVision
        </Typography>
      </Box>

      <Divider />
      <Box
        sx={{
          overflowY: 'auto',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          scrollbarWidth: 'none', // Firefox
          '&::-webkit-scrollbar': {
            display: 'none', // Chrome, Safari, Edge
          },
        }}
      >
        <MenuContent />
      </Box>

      <Stack
        direction="row"
        sx={{
          p: 2,
          gap: 1,
          alignItems: 'center',
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        {/* <Avatar
          sizes="small"
          alt="Riley Carter"
          src="/static/images/avatar/7.jpg"
          sx={{ width: 36, height: 36 }}
        /> */}
        <Box sx={{ mr: 'auto' }}>
          <Typography variant="body2" sx={{ fontWeight: 500, lineHeight: '16px' }}>
            Riley Carter
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            riley@email.com
          </Typography>
        </Box>
        <OptionsMenu />
      </Stack>
    </Drawer>
  );
}

SideMenuMobile.propTypes = {
  open: PropTypes.bool,
  toggleDrawer: PropTypes.func.isRequired,
};

export default SideMenuMobile;
export function CustomIcon() {
  return (
    <Box
      sx={{
        width: '2.2rem',
        height: '2.2rem',
        backgroundColor: '#00205B', // Airbus blue
        borderRadius: '0.5rem', // Rounded corners
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        alignSelf: 'center',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
        padding: '0.25rem',
      }}
    >
      <img
        src="/Airbusicon.svg"
        alt="Aerovision Logo"
        style={{
          width: '80%',
          height: '80%',
          objectFit: 'contain',
          display: 'block', //
           // Makes black to white
        }}
      />

    </Box>
  );
}
