import { Box, Button, styled } from "@mui/material";
import { useNavigate } from "react-router-dom";

// STYLED COMPONENTS
const FlexBox = styled(Box)({
  display: "flex",
  alignItems: "center"
});

const JustifyBox = styled(FlexBox)({
  maxWidth: 320,
  flexDirection: "column",
  justifyContent: "center"
});

const IMG = styled("img")({
  width: "100%",
  marginBottom: "32px"
});

const UnauthorizedRoot = styled(FlexBox)({
  width: "100%",
  alignItems: "center",
  justifyContent: "center",
  height: "100vh !important"
});

export default function Unauthorized() {
  const navigate = useNavigate();

  return (
    <UnauthorizedRoot>
      <JustifyBox>
        <IMG src="/assets/images/illustrations/403.svg" alt="Unauthorized Access" />
        <h2><center>You are not authorized to access this page.</center></h2>
        <Button
          color="primary"
          variant="contained"
          sx={{ textTransform: "capitalize" }}
          onClick={() => navigate("/profile&Settings/profile")}>
          Go Back
        </Button>
      </JustifyBox>
    </UnauthorizedRoot>
  );
}
