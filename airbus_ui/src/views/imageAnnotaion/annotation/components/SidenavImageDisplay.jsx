import { useEffect, useState } from "react";
import { Box, Typography, styled, Fade, CircularProgress } from "@mui/material";
import { Photo } from "@mui/icons-material";
import api from "../../../../utils/UseAxios";

// Updated styled components to match capture theme
const MaxCustomaizer = styled("div")(({ theme }) => ({
  width: "100%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
}));

const HeaderBox = styled(Box)({
  padding: "16px 0 12px 0",
  borderBottom: "2px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "16px",
});

const HeaderTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "600",
  color: "#333",
  textAlign: "center",
});

const StyledScrollContainer = styled("div")(() => ({
  flex: 1,
  overflow: "auto",
  paddingRight: "8px",
  "&::-webkit-scrollbar": {
    width: "6px",
  },
  "&::-webkit-scrollbar-track": {
    background: "rgba(102, 126, 234, 0.1)",
    borderRadius: "3px",
  },
  "&::-webkit-scrollbar-thumb": {
    background: "rgba(102, 126, 234, 0.4)",
    borderRadius: "3px",
    "&:hover": {
      background: "rgba(102, 126, 234, 0.6)",
    },
  },
}));

const ImageBox = styled(Box)(({ isAnnotated, isSelected }) => ({
  width: "100%",
  marginBottom: "12px",
  cursor: "pointer",
  borderRadius: "8px",
  overflow: "hidden",
  position: "relative",
  border: isAnnotated 
    ? "3px solid #4caf50" 
    : isSelected 
      ? "3px solid #667eea" 
      : "2px solid rgba(102, 126, 234, 0.2)",
  backgroundColor: isAnnotated ? "rgba(76, 175, 80, 0.1)" : "white",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  transform: isSelected ? "scale(1.02)" : "scale(1)",
  boxShadow: isSelected 
    ? "0 8px 25px rgba(102, 126, 234, 0.3)" 
    : isAnnotated 
      ? "0 4px 15px rgba(76, 175, 80, 0.2)"
      : "0 2px 8px rgba(0, 0, 0, 0.1)",
  "&:hover": {
    transform: "scale(1.02)",
    boxShadow: "0 8px 25px rgba(102, 126, 234, 0.25)",
    border: isAnnotated 
      ? "3px solid #4caf50"
      : "3px solid #667eea",
  },
}));

const StyledImage = styled("img")({
  width: "100%",
  height: "80px",
  objectFit: "cover",
  display: "block",
  border: "none",
  outline: "none",
});

const ImageOverlay = styled(Box)({
  position: "absolute",
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: "rgba(0, 0, 0, 0.7)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  opacity: 0,
  transition: "opacity 0.3s ease",
  ".image-box:hover &": {
    opacity: 1,
  },
});

const ImageNumber = styled(Box)(({ isAnnotated }) => ({
  position: "absolute",
  top: "6px",
  right: "6px",
  backgroundColor: isAnnotated ? "#4caf50" : "rgba(0, 0, 0, 0.8)",
  color: "white",
  borderRadius: "50%",
  width: "24px",
  height: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: "0.75rem",
  fontWeight: "bold",
  boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
  zIndex: 2,
}));

const AnnotatedBadge = styled(Box)({
  position: "absolute",
  bottom: "6px",
  left: "6px",
  backgroundColor: "#4caf50",
  color: "white",
  borderRadius: "12px",
  padding: "2px 8px",
  fontSize: "0.65rem",
  fontWeight: "600",
  textTransform: "uppercase",
  letterSpacing: "0.5px",
  boxShadow: "0 2px 8px rgba(76, 175, 80, 0.4)",
  zIndex: 2,
});

const EmptyState = styled(Box)({
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#666",
  textAlign: "center",
  padding: "32px 16px",
});

const LoadingState = styled(Box)({
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#666",
  gap: 2,
});

export default function SidenavImageDisplay({
  pieceLabel,
  onImageSelect,
  onFirstImageLoad,
  annotatedImages
}) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');

  useEffect(() => {
    async function fetchImages() {
      if (!pieceLabel) {
        setImages([]);
        return;
      }

      try {
        setLoading(true);
        const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
        const data = response.data;
        setImages(data);

        if (data.length > 0 && onFirstImageLoad) {
          onFirstImageLoad(data[0].url);
          setSelectedImageUrl(data[0].url);
        }
      } catch (error) {
        console.error("Error fetching images:", error.response?.data?.detail || error.message);
        setImages([]);
      } finally {
        setLoading(false);
      }
    }

    fetchImages();
  }, [pieceLabel, onFirstImageLoad]);

  const handleImageClick = (imageUrl) => {
    setSelectedImageUrl(imageUrl);
    onImageSelect(imageUrl);
  };

  return (
    <MaxCustomaizer>
      <HeaderBox>
        <HeaderTitle>
          {pieceLabel ? `${pieceLabel} Images` : "Select a Piece"}
        </HeaderTitle>
        {images.length > 0 && (
          <Typography
            variant="body2"
            sx={{
              color: "#666",
              textAlign: "center",
              mt: 1,
              fontSize: "0.8rem",
            }}
          >
            {images.length} image{images.length !== 1 ? 's' : ''} • {annotatedImages.length} annotated
          </Typography>
        )}
      </HeaderBox>

      {loading ? (
        <LoadingState>
          <CircularProgress sx={{ color: '#667eea' }} size={32} />
          <Typography variant="body2" sx={{ opacity: 0.8, mt: 1 }}>
            Loading images...
          </Typography>
        </LoadingState>
      ) : images.length === 0 ? (
        <EmptyState>
          <Photo sx={{ fontSize: 48, opacity: 0.4, mb: 2 }} />
          <Typography variant="body1" sx={{ mb: 1, opacity: 0.9 }}>
            No Images
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            {pieceLabel ? `No images found for "${pieceLabel}"` : 'Select a piece to view images'}
          </Typography>
        </EmptyState>
      ) : (
        <StyledScrollContainer>
          {images.map((image, index) => {
            const isAnnotated = annotatedImages.includes(image.url);
            const isSelected = selectedImageUrl === image.url;
            
            return (
              <ImageBox
                key={index}
                className="image-box"
                onClick={() => handleImageClick(image.url)}
                isAnnotated={isAnnotated}
                isSelected={isSelected}
              >
                <StyledImage 
                  src={image.url} 
                  alt={`Image ${index + 1}`}
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjgwIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNmNWY1ZjUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyIiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+SW1hZ2Ugbm90IGZvdW5kPC90ZXh0Pjwvc3ZnPg==';
                  }}
                />
                
                <ImageNumber isAnnotated={isAnnotated}>
                  {index + 1}
                </ImageNumber>
                
                {isAnnotated && (
                  <AnnotatedBadge>
                    Done
                  </AnnotatedBadge>
                )}
                
                <ImageOverlay>
                  <Typography
                    variant="body2"
                    sx={{
                      color: "white",
                      fontWeight: "600",
                      textAlign: "center",
                    }}
                  >
                    {isSelected ? "Selected" : "Click to Select"}
                  </Typography>
                </ImageOverlay>
              </ImageBox>
            );
          })}
        </StyledScrollContainer>
      )}
    </MaxCustomaizer>
  );
}