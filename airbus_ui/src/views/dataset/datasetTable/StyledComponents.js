import { styled } from "@mui/material";
import { Box, Card, Table, Typography, Chip, IconButton, Button } from "@mui/material";

// STYLED COMPONENTS
export const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  position: "relative", // Add this for sidebar positioning
}));

export const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-end",
  flexWrap: "wrap",
  gap: "16px",
});

export const Title = styled(Typography)({
  fontSize: "1.5rem",
  fontWeight: "700",
  color: "#333",
  textTransform: "none",
});

export const FilterCard = styled(Card)(({ theme }) => ({
  marginBottom: "24px",
  border: "1px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "12px",
  overflow: "hidden",
  backgroundColor: "#fff",
}));

export const ModernCard = styled(Card)(({ theme }) => ({
  marginBottom: "24px",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  overflow: "hidden",
  backgroundColor: "#fff",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  "&:hover": {
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.15)",
    border: "2px solid rgba(102, 126, 234, 0.2)",
  },
}));

export const ProductTable = styled(Table)(() => ({
  minWidth: 900,
  "& .MuiTableCell-root": {
    borderBottom: "1px solid rgba(102, 126, 234, 0.08)",
    padding: "16px 12px",
    fontSize: "0.875rem",
  },
  "& .MuiTableHead-root .MuiTableCell-root": {
    backgroundColor: "rgba(102, 126, 234, 0.02)",
    fontWeight: "600",
    color: "#667eea",
    textTransform: "uppercase",
    fontSize: "0.75rem",
    letterSpacing: "1px",
  },
  "& .MuiTableRow-root:hover": {
    backgroundColor: "rgba(102, 126, 234, 0.02)",
  },
}));

export const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "28px",
  minWidth: "90px",
  backgroundColor: variant === 'completed' 
    ? "rgba(76, 175, 80, 0.15)" 
    : variant === 'trained'
    ? "rgba(102, 126, 234, 0.15)"
    : "rgba(244, 67, 54, 0.15)",
  color: variant === 'completed' 
    ? "#4caf50" 
    : variant === 'trained'
    ? "#667eea"
    : "#f44336",
  border: `1px solid ${variant === 'completed' 
    ? "rgba(76, 175, 80, 0.3)" 
    : variant === 'trained'
    ? "rgba(102, 126, 234, 0.3)"
    : "rgba(244, 67, 54, 0.3)"}`,
}));

export const ActionButton = styled(IconButton)(({ variant }) => ({
  width: "32px",
  height: "32px",
  margin: "0 2px",
  borderRadius: "6px",
  border: "1px solid rgba(102, 126, 234, 0.2)",
  backgroundColor: "transparent",
  color: variant === 'delete' ? "#f44336" : variant === 'train' ? "#667eea" : "#666",
  transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
  "&:hover": {
    backgroundColor: variant === 'delete' 
      ? "rgba(244, 67, 54, 0.08)" 
      : variant === 'train' 
      ? "rgba(102, 126, 234, 0.08)"
      : "rgba(0, 0, 0, 0.04)",
    borderColor: variant === 'delete' ? "#f44336" : variant === 'train' ? "#667eea" : "#999",
  },
}));

export const TrainButton = styled(Button)(({ theme }) => ({
  textTransform: "none",
  fontSize: "0.875rem",
  marginLeft: theme.spacing(2),
  borderRadius: "8px",
}));

export const StatsContainer = styled(Box)({
  display: "flex",
  gap: "16px",
  marginBottom: "24px",
  flexWrap: "wrap",
});

export const StatCard = styled(Card)({
  padding: "16px",
  minWidth: "140px",
  textAlign: "center",
  border: "1px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "12px",
  backgroundColor: "rgba(102, 126, 234, 0.02)",
});

export const StatValue = styled(Typography)({
  fontSize: "1.8rem",
  fontWeight: "700",
  color: "#667eea",
  lineHeight: 1,
});

export const StatLabel = styled(Typography)({
  fontSize: "0.75rem",
  color: "#666",
  textTransform: "uppercase",
  letterSpacing: "1px",
  marginTop: "4px",
});