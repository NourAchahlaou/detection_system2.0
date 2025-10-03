import * as React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import Breadcrumbs, { breadcrumbsClasses } from '@mui/material/Breadcrumbs';
import NavigateNextRoundedIcon from '@mui/icons-material/NavigateNextRounded';

const StyledBreadcrumbs = styled(Breadcrumbs)(({ theme }) => ({
  margin: theme.spacing(1, 0),
  [`& .${breadcrumbsClasses.separator}`]: {
    color: (theme.vars || theme).palette.action.disabled,
    margin: 1,
  },
  [`& .${breadcrumbsClasses.ol}`]: {
    alignItems: 'center',
  },
}));

const StyledLink = styled(Link)(({ theme }) => ({
  textDecoration: 'none',
  color: theme.palette.text.secondary,
  '&:hover': {
    color: theme.palette.primary.main,
  },
}));

// Define route configurations with breadcrumb labels
const routeConfig = {
  // '/': { label: 'Dashboard', parent: null },
  // '/dashboard': { label: 'Dashboard', parent: null },
  '/captureImage': { label: 'Capture Image', parent: '/PiecesGroupOverview' },
  '/annotation': { label: 'Annotation', parent: '/PiecesGroupOverview' },
  '/piecesOverview': { label: 'Pieces Overview', parent: '/PiecesGroupOverview' },
  '/dataset': { label: 'Manage Dataset', parent: '/PiecesGroupOverview' },
  '/detection': { label: 'Detection', parent: '/PiecesGroupOverview' },
  '/detectionLotsOverview': { label: 'Verify Lot', parent: '/PiecesGroupOverview' },
  '/identification': { label: 'Identify Piece', parent: '/PiecesGroupOverview' },
  '/piecesGroupOverview': { label: 'Group Overview', parent: '/PiecesGroupOverview' },
  '/pieceImageViewer': { label: 'Piece Image Viewer', parent: '/PiecesGroupOverview' },
  '/lotSessionViewer': { label: 'Lot Session', parent: '/PiecesGroupOverview' },
  '/profile': { label: 'Profile', parent: '/PiecesGroupOverview' },
};

export default function NavbarBreadcrumbs() {
  const location = useLocation();

  const generateBreadcrumbs = () => {
    const currentPath = location.pathname;
    const breadcrumbs = [];
    
    // Function to build breadcrumb chain recursively
    const buildChain = (path) => {
      const config = routeConfig[path];
      if (!config) return;
      
      if (config.parent && config.parent !== path) {
        buildChain(config.parent);
      }
      
      breadcrumbs.push({
        path: path,
        label: config.label,
        isLast: path === currentPath
      });
    };

    buildChain(currentPath);
    return breadcrumbs;
  };

  const breadcrumbs = generateBreadcrumbs();

  // Don't render breadcrumbs if we don't have route config or if it's just dashboard
  if (!breadcrumbs.length || (breadcrumbs.length === 1 && breadcrumbs[0].path === '/PiecesGroupOverview')) {
    return null;
  }

  return (
    <StyledBreadcrumbs
      aria-label="breadcrumb"
      separator={<NavigateNextRoundedIcon fontSize="small" />}
    >
      {breadcrumbs.map((crumb, index) => {
        const isLast = index === breadcrumbs.length - 1;
        
        if (isLast) {
          return (
            <Typography 
              key={crumb.path} 
              variant="body1" 
              sx={{ color: 'text.primary', fontWeight: 600 }}
            >
              {crumb.label}
            </Typography>
          );
        }
        
        return (
          <StyledLink key={crumb.path} to={crumb.path}>
            <Typography variant="body1">
              {crumb.label}
            </Typography>
          </StyledLink>
        );
      })}
    </StyledBreadcrumbs>
  );
}