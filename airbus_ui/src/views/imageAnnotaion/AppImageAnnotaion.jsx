import React, { Fragment, useState, useEffect } from "react";
import { Card, Grid, Box, styled } from "@mui/material";
import NonAnnotated from "./NonAnnotated";
import SidenavImageDisplay from "./annotation/components/SidenavImageDisplay";
import Simple from "./Simple";
import api from "../../utils/UseAxios";
import { useNavigate } from "react-router-dom";

// STYLED COMPONENTS - Updated to match capture theme
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));

const ContainerPieces = styled("div")(() => ({
  display: "flex",
  overflowX: "auto",
  paddingBottom: "16px",
  scrollbarWidth: "none",
  "&::-webkit-scrollbar": {
    display: "none",
  },
}));

const ContentBox = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

// Updated main annotation card to match VideoCard styling
const AnnotationCard = styled(Card)(({ theme }) => ({
  width: "100%",
  height: "600px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: "2px solid #667eea",
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  padding: 0,
  margin: 0,
  // Override any Material-UI Card default styles
  '& .MuiCard-root': {
    padding: 0,
  },
  [theme.breakpoints.down("md")]: {
    height: "500px",
  },
  [theme.breakpoints.down("sm")]: {
    height: "400px",
  },
}));

// Updated sidebar card to match theme
const SidebarCard = styled(Card)(({ theme }) => ({
  height: "600px",
  backgroundColor: "#f5f5f5",
  border: "2px solid #667eea",
  borderRadius: "12px",
  padding: "16px",
  margin: 0,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  [theme.breakpoints.down("md")]: {
    height: "500px",
  },
  [theme.breakpoints.down("sm")]: {
    height: "400px",
    // Removed marginTop to keep them in same row
  },
}));

export default function AppImageAnnotaion() {
  const [selectedPieceLabel, setSelectedPieceLabel] = useState('');
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  const [initialPiece, setInitialPiece] = useState(null);
  const [annotatedImages, setAnnotatedImages] = useState([]);
  const navigate = useNavigate();

  const handlePieceSelect = (pieceLabel) => {
    setSelectedPieceLabel(pieceLabel);
  };

  const handleImageSelect = (url) => {
    setSelectedImageUrl(url);
  };

  const handleFirstImageLoad = (url) => {
    setSelectedImageUrl(url);
  };

  useEffect(() => {
    async function fetchInitialPiece() {
      try {
        const response = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const pieces = response.data;
        if (pieces.length > 0) {
          const firstPiece = pieces[0];
          setInitialPiece(firstPiece.piece_label);
          setSelectedPieceLabel(firstPiece.piece_label);
        } else {
          navigate("/204");
        }
      } catch (error) {
        console.error("Error fetching initial piece:", error.response?.data?.detail || error.message);
        navigate("/204");
      }
    }

    fetchInitialPiece();
  }, [navigate]);

  const handleAnnotationSubmit = async () => {
    try {
      await api.post("/api/annotation/annotations", { imageUrl: selectedImageUrl });
      setAnnotatedImages((prev) => [...prev, selectedImageUrl]);
    } catch (error) {
      console.error("Error saving annotation:", error.response?.data?.detail || error.message);
    }
  };

  return (
    <Container>
      <ContainerPieces>
        <NonAnnotated onPieceSelect={handlePieceSelect} />
      </ContainerPieces>
      <Fragment>
        <ContentBox>
          <Box display="flex" height="100%">
            <Grid container spacing={3} style={{ flexGrow: 1 }} wrap="nowrap">
              <Grid item lg={9} md={8} sm={8} xs={8} style={{ display: "flex", flexShrink: 1 }}>
                <AnnotationCard>
                  <Simple 
                    imageUrl={selectedImageUrl} 
                    annotated={annotatedImages.includes(selectedImageUrl)} 
                    pieceLabel={selectedPieceLabel}
                  />
                </AnnotationCard>
              </Grid>
              <Grid item lg={3} md={4} sm={4} xs={4} style={{ display: "flex", flexShrink: 0 }}>
                <SidebarCard>
                  <SidenavImageDisplay
                    pieceLabel={selectedPieceLabel}
                    onImageSelect={handleImageSelect}
                    onFirstImageLoad={handleFirstImageLoad}
                    annotatedImages={annotatedImages}
                  />
                </SidebarCard>
              </Grid>
            </Grid>
          </Box>
        </ContentBox>
      </Fragment>
    </Container>
  );
}