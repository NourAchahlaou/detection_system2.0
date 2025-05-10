import * as React from 'react';
import { 
  Box, 
  Typography, 
  Avatar, 
  Chip, 
  Stack, 
  Divider,
  Card,
  CardContent,
  
  alpha
} from '@mui/material';
import { 
  Schedule as ClockIcon, 
  Notifications as BellIcon, 
  PhotoCamera as CameraIcon, 
  Security as ShieldIcon,
  Email as EmailIcon,

} from '@mui/icons-material';

export default function SideCard() {
  return (
    <Card 
      elevation={2} 
      variant="outlined" 
      sx={{ 
        height: '100%',
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'auto',
        border: '1px solid',
        borderColor: 'divider',
        backgroundColor: (theme) => theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.8) : alpha(theme.palette.background.paper, 1),
      }}
    >
      <CardContent sx={{ 
        p: 3,
        '&:last-child': { pb: 3 },
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>
        {/* Profile Header */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          width: '100%',
          position: 'relative',
          mb: 1
        }}>

          
          <Typography variant="h5" fontWeight="700" sx={{ mb: 0.5 }}>Achahlaou Nour</Typography>
          
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1.5, 
            my: 1,
            flexWrap: 'wrap',
            justifyContent: 'center'
          }}>
            <Chip 
              icon={<CameraIcon fontSize="small" />} 
              label="Technician" 
              color="primary" 
              sx={{
                px: 1,
                fontWeight: 500,
                borderRadius: 1.5
              }}
              size="small"
            />
            
            <Chip
              sx={{ 
                bgcolor: (theme) => alpha(theme.palette.success.main, 0.1),
                color: 'success.dark',
                fontWeight: 600,
                borderRadius: 1.5,
                '& .MuiChip-icon': {
                  color: 'success.main'
                }
              }}
              icon={
                <Box component="span" sx={{ 
                  width: 8, 
                  height: 8, 
                  borderRadius: '50%', 
                  bgcolor: 'success.main',
                  display: 'inline-block'
                }}/>
              }
              iconColor="success"
              label="Active Now"
              size="small"
            />
          </Box>
        </Box>
        
        <Divider sx={{ width: '100%' }} />
        
        {/* User Info Section */}
        <Box sx={{ width: '100%' }}>
          <Stack spacing={2} sx={{ fontSize: '0.875rem' }}>
          {/* TODO : i'll go back to it later  */}
            {/* <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              pb: 1
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <ClockIcon fontSize="small" sx={{ mr: 1.5, color: 'primary.main', fontSize: '1.1rem' }} />
                <Typography variant="body2" color="text.secondary" fontWeight={500}>Working Hours</Typography>
              </Box>
              <Typography variant="body2" fontWeight={600} sx={{ color: 'text.primary' }}>07h 45min</Typography>
            </Box> */}
            
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              pb: 1
            }}>
              <EmailIcon fontSize="small" sx={{ mr: 1.5, color: 'primary.main', fontSize: '1.1rem' }} />
              <Typography 
                variant="body2" 
                color="text.primary" 
                fontWeight={500} 
                noWrap 
                sx={{ 
                  maxWidth: '100%',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}
              >
                nour.achahlaou@airbus.com
              </Typography>
            </Box>
            
      

            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              pb: 1
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <BellIcon fontSize="small" sx={{ mr: 1.5, color: 'primary.main', fontSize: '1.1rem' }} />
                <Typography variant="body2" color="text.secondary" fontWeight={500}>Alerts</Typography>
              </Box>
              <Chip
                label="3"
                color="error"
                size="small"
                sx={{ 
                  fontWeight: 'bold',
                  borderRadius: 1,
                  minWidth: '28px'
                }}
              />
            </Box>
          </Stack>
        </Box>
        
        <Divider sx={{ width: '100%' }} />
        
        {/* Access Level Section */}
        <Box sx={{ width: '100%' }}>
          <Typography 
            variant="subtitle1" 
            fontWeight={600} 
            color="text.primary"
            sx={{ mb: 1.5 }}
          >
            Access Level
          </Typography>
          <Box sx={{ 
            display: 'flex', 
            gap: 2, 
            alignItems: 'center', 
            bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08),
            p: 1.5,
            borderRadius: 1.5
          }}>
            <Avatar sx={{ 
              width: 38, 
              height: 38, 
              bgcolor: 'primary.main'
            }}>
              <ShieldIcon sx={{ fontSize: '1.2rem' }} />
            </Avatar>
            <Box>
              <Typography variant="body1" fontWeight={600} color="text.primary">Level 2</Typography>
              <Typography variant="body2" color="text.secondary">Standard Technician</Typography>
            </Box>
          </Box>
        </Box>
        
        <Divider sx={{ width: '100%' }} />
        
        {/* Work Area Section */}
        <Box sx={{ width: '100%' }}>
          <Typography 
            variant="subtitle1" 
            fontWeight={600} 
            color="text.primary"
            sx={{ mb: 1.5 }}
          >
            Work Area
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            <Chip 
              label="Assembly Line B" 
              variant="outlined" 
              size="medium" 
              color="primary"
              sx={{ 
                fontSize: '0.8rem',
                fontWeight: 500,
                borderRadius: 1.5,
                py: 0.5
              }}
            />
            <Chip 
              label="Section 5" 
              variant="outlined" 
              size="medium" 
              color="primary"
              sx={{ 
                fontSize: '0.8rem',
                fontWeight: 500,
                borderRadius: 1.5,
                py: 0.5
              }}
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}