import React from "react";
import { 
  Box, 
  Button, 
  Typography, 
  styled, 
  Card,
  CardContent,
  Stack,
} from "@mui/material";
import { 
  CameraAlt, 
  ArrowBack,
  DatasetOutlined,
  ErrorOutline
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";


const NoDataRoot = styled(Box)(({ theme }) => ({
  width: "100%",
  minHeight: "100vh",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "#0d1117",
  position: "relative",
  overflow: "hidden",
  "&::before": {
    content: '""',
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: "radial-gradient(circle at 30% 20%, rgba(33, 38, 45, 0.8) 0%, transparent 50%), radial-gradient(circle at 70% 80%, rgba(13, 17, 23, 0.9) 0%, transparent 50%)",
    zIndex: 1,
  }
}));

const ContentCard = styled(Card)(({ theme }) => ({
  maxWidth: 520,
  width: "100%",
  padding: "48px 40px",
  textAlign: "center",
  background: "#161b22",
  border: "1px solid #30363d",
  borderRadius: "12px",
  boxShadow: "0 16px 32px rgba(1, 4, 9, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.04)",
  position: "relative",
  zIndex: 2,
  [theme.breakpoints.down("sm")]: {
    margin: "16px",
    padding: "32px 24px",
  },
}));

const IconContainer = styled(Box)(({ theme }) => ({
  width: 80,
  height: 80,
  borderRadius: "12px",
  background: "#21262d",
  border: "1px solid #30363d",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  margin: "0 auto 24px auto",
  position: "relative",
}));

const ActionButton = styled(Button)(({ theme }) => ({
  borderRadius: "6px",
  padding: "12px 24px",
  fontSize: "14px",
  fontWeight: 600,
  textTransform: "none",
  background: "#238636",
  color: "#ffffff",
  border: "1px solid #238636",
  boxShadow: "0 1px 0 rgba(27, 31, 36, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.25)",
  transition: "all 0.2s cubic-bezier(0.3, 0, 0.5, 1)",
  "&:hover": {
    background: "#2ea043",
    borderColor: "#2ea043",
    transform: "translateY(-1px)",
    boxShadow: "0 4px 8px rgba(35, 134, 54, 0.3)",
  },
  "&:active": {
    background: "#1f7b2f",
    transform: "translateY(0px)",
  }
}));

const SecondaryButton = styled(Button)(({ theme }) => ({
  borderRadius: "6px",
  padding: "12px 24px",
  fontSize: "14px",
  fontWeight: 600,
  textTransform: "none",
  color: "#f0f6fc",
  backgroundColor: "#21262d",
  border: "1px solid #30363d",
  transition: "all 0.2s cubic-bezier(0.3, 0, 0.5, 1)",
  "&:hover": {
    backgroundColor: "#30363d",
    borderColor: "#484f58",
    transform: "translateY(-1px)",
  }
}));

const StatusIndicator = styled(Box)(({ theme }) => ({
  position: "absolute",
  top: "16px",
  right: "16px",
  display: "flex",
  alignItems: "center",
  gap: "6px",
  padding: "4px 12px",
  background: "#da3633",
  color: "#ffffff",
  borderRadius: "16px",
  fontSize: "11px",
  fontWeight: "600",
  textTransform: "uppercase",
  letterSpacing: "0.5px",
  zIndex: 10,
}));

const MetricCard = styled(Box)(({ theme }) => ({
  background: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: "8px",
  padding: "16px",
  textAlign: "center",
  transition: "border-color 0.2s ease",
  "&:hover": {
    borderColor: "#30363d",
  }
}));

const GridPattern = styled(Box)({
  position: "absolute",
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  opacity: 0.02,
  backgroundImage: `
    linear-gradient(rgba(240, 246, 252, 0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(240, 246, 252, 0.1) 1px, transparent 1px)
  `,
  backgroundSize: "24px 24px",
  zIndex: 1,
});

export default function NoData() {
  const navigate = useNavigate();

  return (
    <NoDataRoot>
      <GridPattern />
      
      <ContentCard>
        <StatusIndicator>
          <ErrorOutline sx={{ fontSize: 12 }} />
          NO DATA
        </StatusIndicator>
        
        <CardContent sx={{ padding: 0 }}>
          <IconContainer>
            <DatasetOutlined sx={{ fontSize: 32, color: "#7d8590" }} />
          </IconContainer>
          
          <Typography 
            variant="h4" 
            component="h1"
            sx={{ 
              fontWeight: 600,
              marginBottom: 1,
              color: "#f0f6fc",
              fontSize: "24px",
              fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif",
            }}
          >
            No data available
          </Typography>
          
          <Typography 
            variant="body1" 
            sx={{ 
              color: "#8b949e",
              marginBottom: 4,
              lineHeight: 1.5,
              fontSize: "16px",
              fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif",
            }}
          >
            No artifacts have been captured yet. Start collecting industrial data 
            to populate your workspace and enable analysis capabilities.
          </Typography>

          {/* Metrics Row */}
          <Stack direction="row" spacing={2} sx={{ mb: 4 }}>
            <MetricCard sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ color: "#f0f6fc", fontWeight: 600, mb: 0.5 }}>
                0
              </Typography>
              <Typography variant="caption" sx={{ color: "#8b949e", fontSize: "12px" }}>
                Artifacts
              </Typography>
            </MetricCard>
            <MetricCard sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ color: "#f0f6fc", fontWeight: 600, mb: 0.5 }}>
                0
              </Typography>
              <Typography variant="caption" sx={{ color: "#8b949e", fontSize: "12px" }}>
                Collections
              </Typography>
            </MetricCard>
            <MetricCard sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ color: "#f0f6fc", fontWeight: 600, mb: 0.5 }}>
                0
              </Typography>
              <Typography variant="caption" sx={{ color: "#8b949e", fontSize: "12px" }}>
                Reports
              </Typography>
            </MetricCard>
          </Stack>
          
          <Stack direction="column" spacing={2} alignItems="center">
            <ActionButton
              variant="contained"
              size="large"
              startIcon={<CameraAlt sx={{ fontSize: 18 }} />}
              onClick={() => navigate('/captureImage')}
              fullWidth
              sx={{ maxWidth: 320 }}
            >
              Start data collection
            </ActionButton>
            
            <SecondaryButton
              variant="outlined"
              size="large"
              startIcon={<ArrowBack sx={{ fontSize: 18 }} />}
              onClick={() => navigate('/PiecesGroupOverview')}
              fullWidth
              sx={{ maxWidth: 320 }}
            >
              Back to dashboard
            </SecondaryButton>
          </Stack>
          

        </CardContent>
      </ContentCard>
    </NoDataRoot>
  );
}