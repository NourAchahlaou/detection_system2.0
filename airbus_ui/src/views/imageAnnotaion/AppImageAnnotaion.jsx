import React, { Fragment, useState, useEffect } from "react";
import { Card, Grid, Box, styled } from "@mui/material";
import NonAnnotated from "./NonAnnotated";
import SidenavImageDisplay from "./annotation/components/SidenavImageDisplay";
import Simple from "./Simple";
import axios from "axios";
import { useNavigate } from "react-router-dom"; // Import useNavigate

// STYLED COMPONENTS
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" }
}));

const ContainerPieces = styled("div")(() => ({
  display: "flex",
  overflowX: "auto",
  // whiteSpace: "nowrap",
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
  const [annotatedImages, setAnnotatedImages] = useState([]); // State to keep track of annotated images
  const navigate = useNavigate(); // Initialize navigate

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
        const response = await axios.get("http://127.0.0.1:8000/piece/get_Img_nonAnnotated");
        const pieces = response.data;
        if (pieces.length > 0) {
          const firstPiece = pieces[0];
          setInitialPiece(firstPiece.piece_label);
          setSelectedPieceLabel(firstPiece.piece_label);
        } else {
          navigate("/204"); // Redirect to No Data page if no pieces are found
        }
      } catch (error) {
        console.error("Error fetching initial piece:", error);
        navigate("/204"); // Redirect to No Data page on error
      }
    }

    fetchInitialPiece();
  }, [navigate]);

  const handleAnnotationSubmit = async () => {
    try {
      await axios.post("http://127.0.0.1:8000/annotations", { imageUrl: selectedImageUrl });
      setAnnotatedImages((prev) => [...prev, selectedImageUrl]);
    } catch (error) {
      console.error("Error saving annotation:", error);
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
                  {/* Pass pieceLabel as a prop to the Simple component */}
                  <Simple 
                    imageUrl={selectedImageUrl} 
                    annotated={annotatedImages.includes(selectedImageUrl)} 
                    pieceLabel={selectedPieceLabel}  // Passing piece label to the Simple component
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
