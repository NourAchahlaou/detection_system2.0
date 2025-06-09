import React, { useEffect, useState } from "react";
import { ExpandLess, StarOutline, WebStories, CropFree } from "@mui/icons-material";
import { Card, Fab, Grid, lighten, styled, useTheme } from "@mui/material";
import api from "../../utils/UseAxios"; // Updated import

// STYLED COMPONENTS
const ContentBox = styled("div")(() => ({
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center"
}));

const FabIcon = styled(Fab)(() => ({
  width: "44px !important",
  height: "44px !important",
  boxShadow: "none !important"
}));

const H3 = styled("h3")(() => ({
  margin: 0,
  fontWeight: "500",
  marginLeft: "12px"
}));

const Span = styled("span")(() => ({
  fontSize: "13px",
  marginLeft: "4px"
}));

const IconBox = styled("div")(() => ({
  width: 16,
  height: 16,
  color: "#fff",
  display: "flex",
  overflow: "hidden",
  borderRadius: "300px ",
  justifyContent: "center",
  "& .icon": { fontSize: "14px" }
}));

export default function NonAnnotated({ onPieceSelect }) {
  const { palette } = useTheme();
  const bgError = lighten(palette.primary.main, 0.85);

  const [pieces, setPieces] = useState([]);

  useEffect(() => {
    async function fetchPieces() {
      try {
        const response = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        setPieces(response.data);
      } catch (error) {
        console.error("Error fetching pieces:", error.response?.data?.detail || error.message);
      }
    }

    fetchPieces();
  }, []);

  return (
    <Grid container spacing={3} sx={{ mb: 3, flexWrap: 'nowrap' }}>
      {pieces.map((piece) => (
        <Grid item xs={12} md={6} key={piece.piece_label}>
          <Card elevation={3} sx={{ p: 2 }} onClick={() => onPieceSelect(piece.piece_label)}>
            <ContentBox>
              <FabIcon size="medium" sx={{ backgroundColor: bgError, overflow: "hidden" }}>
                <CropFree color="primary" />
              </FabIcon>
               
              <H3 color="primary.main">{piece.piece_label}</H3>
            </ContentBox>
             
            <ContentBox sx={{ pt: 2 }}>
              <img
                src={piece.url}
                alt={piece.piece_label}
                style={{
                  maxWidth: "20%",
                  height: "auto",
                  marginLeft: "12px",
                  marginRight: "auto",
                  flexGrow: 1,
                }}
              />
              <IconBox sx={{ backgroundColor: "primary.main" }}>
                <WebStories className="icon" />
              </IconBox>
               
              <Span color="error.main">{piece.nbr_img} images</Span>
            </ContentBox>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}