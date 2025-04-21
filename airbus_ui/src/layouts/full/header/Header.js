import * as React from 'react';
import Stack from '@mui/material/Stack';
import NotificationsRoundedIcon from '@mui/icons-material/NotificationsRounded';
import CustomDatePicker from './CustomDatePicker';
import NavbarBreadcrumbs from '../navbar/NavbarBreadcrumbs';
import MenuButton from '../sidebar/MenuButton';
import ColorModeIconDropdown from '../../../shared-theme/ColorModeIconDropdown';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import SideMenuMobile from '../sidebar/SideMenuMobile';

import Search from './Search';

export default function Header() {
    const [open, setOpen] = React.useState(false);
  
    const toggleDrawer = (newOpen) => () => {
      setOpen(newOpen);
    };
  return (
    <Stack
      direction="row"
      sx={{
        display: { xs: 'none', md: 'flex' },
        width: '100%',
        alignItems: { xs: 'flex-start', md: 'center' },
        justifyContent: 'space-between',
        maxWidth: { sm: '100%', md: '1700px' },
        pt: 1.5,
      }}
      spacing={2}
    >
      <Stack direction="row" sx={{ gap: 1 }}>
        <MenuButton aria-label="menu" onClick={toggleDrawer(true)}>
        <MenuRoundedIcon />
        </MenuButton>
        <SideMenuMobile open={open} toggleDrawer={toggleDrawer} />
        <NavbarBreadcrumbs />
      </Stack>

      <Stack direction="row" sx={{ gap: 1 }}>
        <Search />
        <CustomDatePicker />
        <MenuButton showBadge aria-label="Open notifications">
          <NotificationsRoundedIcon />
        </MenuButton>
        <ColorModeIconDropdown />
      </Stack>
    </Stack>
  );
}
