import React, { Fragment, useState, useEffect } from "react";
import { Card, Grid, Box, styled } from "@mui/material";
import NonAnnotated from "./NonAnnotated";
import SidenavImageDisplay from "./annotation/components/SidenavImageDisplay";
import Simple from "./Simple";
import api from "../../utils/UseAxios"; // Updated import
import { useNavigate } from "react-router-dom";

// STYLED COMPONENTS
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" }
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
            <Grid container spacing={2} style={{ flexGrow: 1 }}>
              <Grid item lg={9} md={9} sm={12} xs={12} style={{ display: "flex" }}>
                <Card
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                  sx={{ mb: 3, flexGrow: 1 }}
                >
                  <Simple 
                    imageUrl={selectedImageUrl} 
                    annotated={annotatedImages.includes(selectedImageUrl)} 
                    pieceLabel={selectedPieceLabel}
                  />
                </Card>
              </Grid>
              <Grid item lg={3} md={3} sm={12} xs={12} style={{ display: "flex" }}>
                <Card sx={{ px: 3, py: 2, mb: 3, flexGrow: 1 }}>
                  <SidenavImageDisplay
                    pieceLabel={selectedPieceLabel}
                    onImageSelect={handleImageSelect}
                    onFirstImageLoad={handleFirstImageLoad}
                    annotatedImages={annotatedImages}
                  />
                </Card>
              </Grid>
            </Grid>
          </Box>
        </ContentBox>
      </Fragment>
    </Container>
  );
}