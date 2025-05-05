import * as React from 'react';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import AutoFixHighRoundedIcon from '@mui/icons-material/AutoFixHighRounded';
import ConstructionRoundedIcon from '@mui/icons-material/ConstructionRounded';
import SettingsSuggestRoundedIcon from '@mui/icons-material/SettingsSuggestRounded';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';
import { ReactComponent as AirVisionLogo } from '../../../assets/Airvisionlogo_updated.svg';

const items = [
  {
    icon: <SettingsSuggestRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Real-time Analysis',
    description:
      'Quick and reliable detection of industrial pieces during operation, ensuring accuracy without delay.',
  },
  {
    icon: <ConstructionRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Robust Industrial Design',
    description:
      'Built to operate in demanding environments with minimal maintenance and high reliability.',
  },
  {
    icon: <ThumbUpAltRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Operator-Friendly',
    description:
      'Simple interface for technicians and admins to monitor, inspect, and control the inspection process.',
  },
  {
    icon: <AutoFixHighRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Intelligent Detection',
    description:
      'Advanced AI model identifies mismatched or defective parts, reducing human error and saving time.',
  },
];

export default function Content() {
  return (
    <Stack
      sx={{ flexDirection: 'column', alignSelf: 'center', gap: 4, maxWidth: 450 }}
    >
      <Box sx={{ display: {xs :'none',md :'flex'},color: '#00205B', marginBottom: -9 , marginTop:-10}}> {/* Optional: Add marginBottom for closer content */}
        <AirVisionLogo width={150} height={200} />
      </Box>

      {items.map((item, index) => (
        <Stack key={index} direction="row" sx={{ gap: 2 }}>
          {item.icon}
          <div>
            <Typography gutterBottom sx={{ fontWeight: 'medium' }}>
              {item.title}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              {item.description}
            </Typography>
          </div>
        </Stack>
      ))}
    </Stack>
  );
}
