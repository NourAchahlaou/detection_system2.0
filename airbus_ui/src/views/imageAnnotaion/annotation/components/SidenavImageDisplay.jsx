import { useEffect, useState } from "react";
import { Box, Card, styled } from "@mui/material";
import BadgeSelected from "../../../../components/theme/customizations/BadgeSelected";

const MaxCustomaizer = styled("div")(({ theme }) => ({
  width: "auto",
  right: "10px",
  display: "flex",
  height: "75vh",
  paddingBottom: "32px",
  flexDirection: "column",
}));

const LayoutBox = styled(BadgeSelected)(({ isAnnotated }) => ({
  width: "100%",
  height: "100px !important",
  cursor: "pointer",
  marginTop: "12px",
  marginBottom: "12px",
  position: "relative",
  backgroundColor: isAnnotated ? "#e0ffe0" : "transparent", // Light green for annotated
  border: isAnnotated ? "2px solid green" : "none", // Border for annotated
  "& .layout-name": { display: "none" },
  "&:hover .layout-name": {
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    position: "absolute",
    justifyContent: "center",
  },
}));

const IMG = styled("img")(() => ({ width: "100%" }));

const StyledScrollContainer = styled("div")(() => ({
  paddingLeft: "16px",
  paddingRight: "16px",
  overflow: "auto",
  height: "100%",
  // Custom scrollbar styling (optional)
  "&::-webkit-scrollbar": {
    width: "8px",
  },
  "&::-webkit-scrollbar-track": {
    background: "#f1f1f1",
    borderRadius: "4px",
  },
  "&::-webkit-scrollbar-thumb": {
    background: "#888",
    borderRadius: "4px",
  },
  "&::-webkit-scrollbar-thumb:hover": {
    background: "#555",
  },
}));

export default function SidenavImageDisplay({ 
  pieceLabel, 
  onImageSelect, 
  onFirstImageLoad, 
  annotatedImages 
}) {
  const [images, setImages] = useState([]);

  useEffect(() => {
    async function fetchImages() {
      if (!pieceLabel) return;

      try {
        const response = await fetch(
          `http://127.0.0.1:8000/piece/get_images_of_piece/${pieceLabel}`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch images");
        }
        const data = await response.json();
        setImages(data);

        if (data.length > 0 && onFirstImageLoad) {
          onFirstImageLoad(data[0].url);
        }
      } catch (error) {
        console.error("Error fetching images:", error);
      }
    }

    fetchImages();
  }, [pieceLabel]);

  return (
    <MaxCustomaizer>
      <Card>
        <StyledScrollContainer>
          {images.map((image, index) => (
            <LayoutBox
              key={index}
              onClick={() => onImageSelect(image.url)}
              isAnnotated={annotatedImages.includes(image.url)}
            >
              <IMG src={image.url} alt={`Image ${index + 1}`} />
              <Box className="layout-name">
                {index + 1}
              </Box>
            </LayoutBox>
          ))}
        </StyledScrollContainer>
      </Card>
    </MaxCustomaizer>
  );
}