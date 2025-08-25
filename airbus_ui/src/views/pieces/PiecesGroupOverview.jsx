import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  styled, 
  Card,
  Chip,
  CircularProgress,
  Button,
  Collapse,
  IconButton,
  Pagination
} from "@mui/material";
import { 
  CropFree, 
  PhotoLibrary, 
  Visibility,
  ExpandMore,
  ExpandLess,
  FolderOpen
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import api from "../../utils/UseAxios";

// STYLED COMPONENTS - Following your theme
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  textAlign: "center",
});

const GroupSection = styled(Box)(({ theme }) => ({
  marginBottom: "32px",
  border: "1px solid rgba(102, 126, 234, 0.15)",
  borderRadius: "16px",
  overflow: "hidden",
  backgroundColor: "#fff",
  boxShadow: "0 4px 16px rgba(0, 0, 0, 0.08)",
}));

const GroupHeader = styled(Box)(({ theme }) => ({
  padding: "20px 24px",
  backgroundColor: "rgba(102, 126, 234, 0.05)",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  cursor: "pointer",
  transition: "all 0.3s ease",
  "&:hover": {
    backgroundColor: "rgba(102, 126, 234, 0.1)",
  },
}));

const GroupTitle = styled(Typography)({
  fontSize: "1.3rem",
  fontWeight: "700",
  color: "#333",
  display: "flex",
  alignItems: "center",
  gap: "12px",
});

const GroupStats = styled(Box)({
  display: "flex",
  alignItems: "center",
  gap: "16px",
});

const GroupContent = styled(Box)({
  padding: "24px",
});

// Pagination controls for groups
const GroupPagination = styled(Box)({
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "16px 24px",
  backgroundColor: "rgba(102, 126, 234, 0.03)",
  borderTop: "1px solid rgba(102, 126, 234, 0.1)",
});

// Grid for pieces within a group
const PiecesGridContainer = styled('div')(({ theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
  gap: '20px',
  width: '100%',
  
  [theme.breakpoints.down('sm')]: {
    gridTemplateColumns: '1fr',
  },
}));

const PieceCard = styled(Card)(({ theme }) => ({
  padding: "20px",
  cursor: "pointer",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  position: "relative",
  overflow: "hidden",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  minWidth: "0",
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "rgba(78, 105, 221, 0.45)",
  },
}));

const CardHeader = styled(Box)({
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between",
  marginBottom: "16px",
  gap: "12px",
});

const IconContainer = styled(Box)({
  width: "48px",
  height: "48px",
  borderRadius: "12px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(102, 126, 234, 0.15)",
  color: "#667eea",
  marginBottom: "12px",
});

const PieceTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "700",
  color: "#333",
  marginBottom: "8px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  "&:hover": {
    color: "#e2e2e2",
  },
});

const StatsContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  marginTop: "auto",
});

const ImagePreview = styled("img")({
  width: "40px",
  height: "40px",
  objectFit: "cover",
  borderRadius: "6px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
});

const ActionButton = styled(Button)({
  textTransform: "none",
  fontWeight: "600",
  borderRadius: "6px",
  fontSize: "0.8rem",
  padding: "4px 12px",
  minWidth: "auto",
  backgroundColor: "#667eea",
  color: "white",
  border: "none",
  "&:hover": {
    backgroundColor: "#5a67d8",
  },
});

const LoadingContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
});

const EmptyState = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
  textAlign: "center",
});

// Constants for pagination
const PIECES_PER_PAGE = 12;
const GROUPS_PER_PAGE = 5;

export default function PiecesGroupOverview() {
  const [allPieces, setAllPieces] = useState([]);
  const [groupedPieces, setGroupedPieces] = useState({});
  const [loading, setLoading] = useState(true);
  const [expandedGroups, setExpandedGroups] = useState({});
  const [groupPages, setGroupPages] = useState({});
  const [currentGroupPage, setCurrentGroupPage] = useState(1);
  const [totalGroups, setTotalGroups] = useState(0);
  const [stats, setStats] = useState({
    totalPieces: 0,
    totalGroups: 0,
    totalImages: 0
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchAllPieces();
  }, []);

  const fetchAllPieces = async () => {
    try {
      setLoading(true);
      
      const response = await api.get("/api/annotation/annotations/get_all_pieces");
      const piecesData = response.data || [];
      
      console.log("Fetched pieces data:", piecesData);
      
      setAllPieces(piecesData);
      
      // Group pieces by prefix or pattern
      const grouped = groupPiecesByPattern(piecesData);
      setGroupedPieces(grouped);
      
      // Initialize pagination for each group
      const initialPages = {};
      Object.keys(grouped).forEach(groupName => {
        initialPages[groupName] = 1;
      });
      setGroupPages(initialPages);
      
      // Calculate stats
      const totalImages = piecesData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
      setStats({
        totalPieces: piecesData.length,
        totalGroups: Object.keys(grouped).length,
        totalImages: totalImages
      });
      
      setTotalGroups(Object.keys(grouped).length);
      
    } catch (error) {
      console.error("Error fetching pieces:", error);
      
      // Fallback logic (same as original)
      try {
        console.log("Trying fallback endpoint...");
        const fallbackResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const nonAnnotatedData = fallbackResponse.data || [];
        
        const convertedData = nonAnnotatedData.map(piece => ({
          piece_label: piece.piece_label,
          nbr_img: piece.nbr_img,
          url: piece.url,
        }));
        
        setAllPieces(convertedData);
        
        const grouped = groupPiecesByPattern(convertedData);
        setGroupedPieces(grouped);
        
        const totalImages = convertedData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
        setStats({
          totalPieces: convertedData.length,
          totalGroups: Object.keys(grouped).length,
          totalImages: totalImages
        });
        
      } catch (fallbackError) {
        console.error("Fallback fetch also failed:", fallbackError);
        setAllPieces([]);
        setGroupedPieces({});
        setStats({ totalPieces: 0, totalGroups: 0, totalImages: 0 });
      }
    } finally {
      setLoading(false);
    }
  };

  // Function to group pieces by common patterns
  const groupPiecesByPattern = (pieces) => {
    const groups = {};
    
    pieces.forEach(piece => {
      // Extract group name from piece label
      // You can modify this logic based on your naming pattern
      let groupName = extractGroupName(piece.piece_label);
      
      if (!groups[groupName]) {
        groups[groupName] = [];
      }
      groups[groupName].push(piece);
    });
    
    return groups;
  };

  // Extract group name from piece label - customize this based on your naming convention
  const extractGroupName = (pieceLabel) => {
    // Example patterns you can customize:
    
    // Pattern 1: Extract prefix before first underscore or dash
    const prefixMatch = pieceLabel.match(/^([^_-]+)/);
    if (prefixMatch) {
      return prefixMatch[1];
    }
    
    // Pattern 2: Extract everything before numbers
    const beforeNumbersMatch = pieceLabel.match(/^([^\d]+)/);
    if (beforeNumbersMatch) {
      return beforeNumbersMatch[1].replace(/[-_\s]+$/, '');
    }
    
    // Pattern 3: Group by first word
    const firstWordMatch = pieceLabel.match(/^(\w+)/);
    if (firstWordMatch) {
      return firstWordMatch[1];
    }
    
    // Fallback: use first 3 characters
    return pieceLabel.substring(0, 3) || 'Other';
  };

  const handleGroupToggle = (groupName) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupName]: !prev[groupName]
    }));
  };

  const handlePieceClick = (pieceLabel) => {
    navigate(`/piece-viewer?piece=${encodeURIComponent(pieceLabel)}`);
  };

  const handleGroupPageChange = (groupName, page) => {
    setGroupPages(prev => ({
      ...prev,
      [groupName]: page
    }));
  };

  const renderPieceCard = (piece) => {
    return (
      <PieceCard key={piece.piece_label} elevation={0} onClick={() => handlePieceClick(piece.piece_label)}>
        <CardHeader>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1, minWidth: 0 }}>
            <IconContainer>
              <CropFree fontSize="medium" />
            </IconContainer>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              <PieceTitle title={piece.piece_label}>{piece.piece_label}</PieceTitle>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <PhotoLibrary sx={{ fontSize: 14, color: "#667eea" }} />
                <Typography variant="caption" sx={{ color: "#666", fontWeight: "500" }}>
                  {piece.nbr_img} images
                </Typography>
              </Box>
            </Box>
          </Box>
          
          {piece.url && (
            <ImagePreview
              src={piece.url}
              alt={piece.piece_label}
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
          )}
        </CardHeader>
        
        <StatsContainer>
          <ActionButton
            size="small"
            startIcon={<Visibility />}
          >
            View Images
          </ActionButton>
        </StatsContainer>
      </PieceCard>
    );
  };

  const renderGroup = (groupName, pieces) => {
    const isExpanded = expandedGroups[groupName] || false;
    const currentPage = groupPages[groupName] || 1;
    const totalPages = Math.ceil(pieces.length / PIECES_PER_PAGE);
    const startIndex = (currentPage - 1) * PIECES_PER_PAGE;
    const paginatedPieces = pieces.slice(startIndex, startIndex + PIECES_PER_PAGE);
    const totalImages = pieces.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);

    return (
      <GroupSection key={groupName}>
        <GroupHeader onClick={() => handleGroupToggle(groupName)}>
          <GroupTitle>
            <FolderOpen />
            {groupName}
          </GroupTitle>
          
          <GroupStats>
            <Chip
              label={`${pieces.length} pieces`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.15)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            <Chip
              label={`${totalImages} images`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.15)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            <IconButton size="small">
              {isExpanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </GroupStats>
        </GroupHeader>
        
        <Collapse in={isExpanded}>
          <GroupContent>
            <PiecesGridContainer>
              {paginatedPieces.map(renderPieceCard)}
            </PiecesGridContainer>
            
            {totalPages > 1 && (
              <GroupPagination>
                <Typography variant="body2" sx={{ color: "#666" }}>
                  Showing {startIndex + 1}-{Math.min(startIndex + PIECES_PER_PAGE, pieces.length)} of {pieces.length} pieces
                </Typography>
                <Pagination
                  count={totalPages}
                  page={currentPage}
                  onChange={(event, page) => handleGroupPageChange(groupName, page)}
                  size="small"
                  color="primary"
                />
              </GroupPagination>
            )}
          </GroupContent>
        </Collapse>
      </GroupSection>
    );
  };

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          <CircularProgress sx={{ color: '#667eea' }} size={48} />
          <Typography variant="h6" sx={{ opacity: 0.8, mt: 2 }}>
            Loading pieces...
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  // Paginate groups
  const groupNames = Object.keys(groupedPieces);
  const totalGroupPages = Math.ceil(groupNames.length / GROUPS_PER_PAGE);
  const startGroupIndex = (currentGroupPage - 1) * GROUPS_PER_PAGE;
  const currentGroups = groupNames.slice(startGroupIndex, startGroupIndex + GROUPS_PER_PAGE);

  return (
    <Container>
      <HeaderBox>       
        <Box sx={{ 
          display: 'flex', 
          gap: 3, 
          mt: 3, 
          flexWrap: 'wrap',
          justifyContent: 'center'
        }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#667eea', fontWeight: '700' }}>
              {stats.totalGroups}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Groups
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#4caf50', fontWeight: '700' }}>
              {stats.totalPieces}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Pieces
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#ff9800', fontWeight: '700' }}>
              {stats.totalImages}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Images
            </Typography>
          </Box>
        </Box>
      </HeaderBox>

      {/* Groups Display */}
      {currentGroups.length > 0 ? (
        <>
          {currentGroups.map(groupName => 
            renderGroup(groupName, groupedPieces[groupName])
          )}
          
          {/* Main Groups Pagination */}
          {totalGroupPages > 1 && (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              mt: 4,
              p: 2,
              backgroundColor: 'rgba(102, 126, 234, 0.05)',
              borderRadius: '12px'
            }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" sx={{ color: "#666" }}>
                  Showing groups {startGroupIndex + 1}-{Math.min(startGroupIndex + GROUPS_PER_PAGE, groupNames.length)} of {groupNames.length}
                </Typography>
                <Pagination
                  count={totalGroupPages}
                  page={currentGroupPage}
                  onChange={(event, page) => setCurrentGroupPage(page)}
                  color="primary"
                />
              </Box>
            </Box>
          )}
        </>
      ) : (
        <EmptyState>
          <FolderOpen sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
          <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
            No Pieces Found
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            No pieces are available in the system
          </Typography>
        </EmptyState>
      )}
    </Container>
  );
}