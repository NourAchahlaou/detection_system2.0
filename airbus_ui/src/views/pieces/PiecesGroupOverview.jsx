import React, { useState, useEffect, useMemo } from "react";
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
  Pagination,
  TextField,
  InputAdornment
} from "@mui/material";
import OutlinedInput from '@mui/material/OutlinedInput';
import FormControl from '@mui/material/FormControl';
import { 
  CropFree, 
  PhotoLibrary, 
  Visibility,
  ExpandMore,
  ExpandLess,
  FolderOpen,
  Search,
  Clear
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import api from "../../utils/UseAxios";
import { datasetService } from "../dataset/datasetService";

// STYLED COMPONENTS - Updated with transparent group sections
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

// Search Section
const SearchSection = styled(Box)(({ theme }) => ({
  marginBottom: "24px",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
}));

const SearchField = styled(TextField)(({ theme }) => ({
  maxWidth: "500px",
  width: "100%",
  "& .MuiOutlinedInput-root": {
    borderRadius: "12px",
    backgroundColor: "rgba(255, 255, 255, 0.8)",
    backdropFilter: "blur(10px)",
    border: "1px solid rgba(102, 126, 234, 0.2)",
    "&:hover": {
      border: "1px solid rgba(102, 126, 234, 0.4)",
    },
    "&.Mui-focused": {
      border: "1px solid #667eea",
      boxShadow: "0 0 0 3px rgba(102, 126, 234, 0.1)",
    },
  },
}));

// Updated GroupSection with transparency
const GroupSection = styled(Box)(({ theme }) => ({
  marginBottom: "32px",
  border: "1px solid rgba(102, 126, 234, 0.15)",
  borderRadius: "16px",
  overflow: "hidden",

}));

const GroupHeader = styled(Box)(({ theme }) => ({
  padding: "20px 24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.15)",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  cursor: "pointer",
  transition: "all 0.3s ease",
  "&:hover": {
    backgroundColor: "rgba(102, 126, 234, 0.15)",
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
  backgroundColor: "rgba(102, 126, 234, 0.05)",
  borderTop: "1px solid rgba(102, 126, 234, 0.1)",
});

// CSS Grid with exactly 4 columns and auto rows
const CardsGridContainer = styled('div')(({ theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(4, 1fr)',
  gap: '20px',
  width: '100%',
  
  [theme.breakpoints.down('lg')]: {
    gridTemplateColumns: 'repeat(3, 1fr)',
  },
  
  [theme.breakpoints.down('md')]: {
    gridTemplateColumns: 'repeat(2, 1fr)',
  },
  
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
  backgroundColor: "rgba(255, 255, 255, 0.9)",
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "rgba(35, 46, 99, 0.81)",
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

const StatsRow = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
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
  const [searchQuery, setSearchQuery] = useState("");
  const [stats, setStats] = useState({
    totalPieces: 0,
    totalGroups: 0,
    totalImages: 0,
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchGroupsAndPieces();
  }, []);

  const fetchGroupsAndPieces = async () => {
    try {
      setLoading(true);
      
      let groupsData = [];
      let piecesData = [];
      
      // Fetch groups from the new backend API
      try {
        console.log("Attempting to fetch groups from:", "/api/artifact_keeper/datasetManager/groups");
        const groupsResponse = await api.get("/api/artifact_keeper/datasetManager/groups");
        console.log("Groups API response:", groupsResponse);
        console.log("Groups response data:", groupsResponse.data);
        
        groupsData = groupsResponse.data?.groups || [];
        console.log("Extracted groups data:", groupsData);
        
        if (!groupsData || groupsData.length === 0) {
          console.warn("No groups found in API response, groupsData:", groupsData);
        }
        
      } catch (groupsError) {
        console.error("Error fetching groups from API:", groupsError);
        console.error("Groups API error details:", {
          message: groupsError.message,
          response: groupsError.response?.data,
          status: groupsError.response?.status
        });
      }
      
      // Fetch all pieces data using the enhanced dataset service
      try {
        console.log("Attempting to fetch pieces using enhanced dataset service...");
        const enhancedData = await datasetService.getAllDatasetsWithFilters({
          search: "",
          page: 1,
          per_page: 1000,
          sort_by: "piece_label",
          sort_order: "asc"
        });
        
        console.log("Enhanced dataset service response:", enhancedData);
        
        if (enhancedData?.data?.pieces) {
          piecesData = enhancedData.data.pieces.map(piece => ({
            piece_label: piece.piece_label,
            nbr_img: piece.total_images || 0,
            url: piece.sample_image_url || piece.url,
            group_label: piece.group_label
          }));
          console.log("Transformed pieces data:", piecesData);
        } else {
          throw new Error("No pieces data in enhanced response");
        }
        
      } catch (enhancedError) {
        console.warn("Enhanced dataset service failed, falling back to original API:", enhancedError);
        
        // Fallback to original API
        try {
          console.log("Attempting to fetch pieces from:", "/api/annotation/annotations/get_all_pieces");
          const piecesResponse = await api.get("/api/annotation/annotations/get_all_pieces");
          piecesData = piecesResponse.data || [];
          console.log("Fetched pieces data (fallback):", piecesData);
          
        } catch (piecesError) {
          console.error("Error fetching pieces:", piecesError);
          throw piecesError;
        }
      }
      
      setAllPieces(piecesData);
      
      // Group pieces by the backend-provided groups
      console.log("Grouping pieces. Groups available:", groupsData.length);
      const grouped = groupPiecesByBackendGroups(piecesData, groupsData);
      console.log("Grouped pieces result:", grouped);
      setGroupedPieces(grouped);
      
      // Initialize pagination for each group
      const initialPages = {};
      Object.keys(grouped).forEach(groupName => {
        initialPages[groupName] = 1;
      });
      setGroupPages(initialPages);
      
      // Calculate stats
      const totalPieces = piecesData.length;
      const totalImages = piecesData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
      
      setStats({
        totalPieces: totalPieces,
        totalGroups: Object.keys(grouped).length,
        totalImages: totalImages,
      });
      
      setTotalGroups(Object.keys(grouped).length);
      
    } catch (error) {
      console.error("Error in fetchGroupsAndPieces:", error);
      
      // Fallback logic
      try {
        console.log("Trying fallback pieces endpoint...");
        const fallbackResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const nonAnnotatedData = fallbackResponse.data || [];
        
        const convertedData = nonAnnotatedData.map(piece => ({
          piece_label: piece.piece_label,
          nbr_img: piece.nbr_img,
          url: piece.url,
        }));
        
        setAllPieces(convertedData);
        
        const grouped = { "Other": convertedData };
        setGroupedPieces(grouped);
        
        const totalImages = convertedData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
        
        setStats({
          totalPieces: convertedData.length,
          totalGroups: 1,
          totalImages: totalImages,
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

  const groupPiecesByBackendGroups = (pieces, groups) => {
    console.log("groupPiecesByBackendGroups called with:", { pieces: pieces.length, groups });
    
    const grouped = {};
    
    if (!groups || groups.length === 0) {
      console.log("No backend groups available, creating fallback groups based on piece patterns");
      
      pieces.forEach((piece) => {
        const pieceLabel = piece.piece_label || "";
        const groupName = piece.group_label || pieceLabel.split('.')[0] || 'Other';
        
        if (!grouped[groupName]) {
          grouped[groupName] = [];
        }
        grouped[groupName].push(piece);
      });
      
      console.log("Fallback groups created:", Object.keys(grouped));
      return grouped;
    }
    
    groups.forEach(groupName => {
      grouped[groupName] = [];
    });
    
    grouped["Other"] = [];
    
    pieces.forEach((piece) => {
      const pieceLabel = piece.piece_label || "";
      const groupLabel = piece.group_label || "";
      let assigned = false;
      
      if (groupLabel && groups.includes(groupLabel)) {
        grouped[groupLabel].push(piece);
        assigned = true;
        console.log(`Assigned piece ${pieceLabel} to group ${groupLabel} (by group_label)`);
      } else {
        for (const groupName of groups) {
          if (pieceLabel.toUpperCase().startsWith(groupName.toUpperCase()) || 
              pieceLabel.toUpperCase().includes(groupName.toUpperCase())) {
            grouped[groupName].push(piece);
            assigned = true;
            console.log(`Assigned piece ${pieceLabel} to group ${groupName} (by pattern)`);
            break;
          }
        }
      }
      
      if (!assigned) {
        grouped["Other"].push(piece);
        console.log(`Assigned piece ${pieceLabel} to Other group`);
      }
    });
    
    Object.keys(grouped).forEach(groupName => {
      if (grouped[groupName].length === 0) {
        console.log(`Removing empty group: ${groupName}`);
        delete grouped[groupName];
      }
    });
    
    console.log("Final grouped result:", Object.keys(grouped).map(key => ({ [key]: grouped[key].length })));
    return grouped;
  };

  // Filter pieces based on search query
  const filteredGroupedPieces = useMemo(() => {
    if (!searchQuery.trim()) {
      return groupedPieces;
    }

    const filtered = {};
    const query = searchQuery.toLowerCase();

    Object.entries(groupedPieces).forEach(([groupName, pieces]) => {
      const filteredPieces = pieces.filter(piece => 
        piece.piece_label.toLowerCase().includes(query)
      );
      
      if (filteredPieces.length > 0) {
        filtered[groupName] = filteredPieces;
      }
    });

    return filtered;
  }, [groupedPieces, searchQuery]);

  const handleGroupToggle = (groupName) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupName]: !prev[groupName]
    }));
  };

  const handlePieceClick = (pieceLabel) => {
    navigate(`/pieceImageViewer?piece=${encodeURIComponent(pieceLabel)}`);
  };

  const handleGroupPageChange = (groupName, page) => {
    setGroupPages(prev => ({
      ...prev,
      [groupName]: page
    }));
  };

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
    const resetPages = {};
    Object.keys(groupedPieces).forEach(groupName => {
      resetPages[groupName] = 1;
    });
    setGroupPages(resetPages);
    setCurrentGroupPage(1);
  };

  const handleClearSearch = () => {
    setSearchQuery("");
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
                console.log(`Failed to load image: ${piece.url}`);
                e.target.style.display = 'none';
              }}
              onLoad={() => {
                console.log(`Successfully loaded image: ${piece.url}`);
              }}
            />
          )}
        </CardHeader>
        
        <StatsContainer>
          <StatsRow>
            <Chip
              label={`${piece.nbr_img} images`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.15)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            
            <ActionButton
              size="small"
              startIcon={<Visibility />}
            >
              View
            </ActionButton>
          </StatsRow>
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
      <GroupSection key={groupName} variant="outlined">
        <GroupHeader onClick={() => handleGroupToggle(groupName)} variant="outlined">
          <GroupTitle>
            <FolderOpen />
            {groupName}
          </GroupTitle>
          
          <GroupStats>
            <Chip
              label={`${pieces.length} pieces`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.2)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            <Chip
              label={`${totalImages} images`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.2)", 
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
            <CardsGridContainer>
              {paginatedPieces.map(renderPieceCard)}
            </CardsGridContainer>
            
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
            Loading groups and pieces...
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  // Paginate groups
  const groupNames = Object.keys(filteredGroupedPieces);
  const totalGroupPages = Math.ceil(groupNames.length / GROUPS_PER_PAGE);
  const startGroupIndex = (currentGroupPage - 1) * GROUPS_PER_PAGE;
  const currentGroups = groupNames.slice(startGroupIndex, startGroupIndex + GROUPS_PER_PAGE);

  return (
    <Container>
      {/* Header with simplified stats */}
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
              Total Groups
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
            <Typography variant="h4" sx={{ color: '#9c27b0', fontWeight: '700' }}>
              {stats.totalImages}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Images
            </Typography>
          </Box>
        </Box>
      </HeaderBox>

      {/* Search Section */}
    <SearchSection >
      <OutlinedInput
          placeholder="Search pieces by name..."
          value={searchQuery}
          onChange={handleSearchChange}
          variant="outlined"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ color: "#667eea" }} />
              </InputAdornment>
            ),
            endAdornment: searchQuery && (
              <InputAdornment position="end">
                <IconButton
                  size="small"
                  onClick={handleClearSearch}
                  sx={{ color: "#667eea" }}
                >
                  <Clear />
                </IconButton>
              </InputAdornment>
            ),
          }}
      />
    </SearchSection>

      {/* Search Results Info */}
      {searchQuery && (
        <Box sx={{ 
          textAlign: 'center', 
          mb: 3,
          p: 2,
          backgroundColor: 'rgba(102, 126, 234, 0.08)',
          borderRadius: '12px',
          border: '1px solid rgba(102, 126, 234, 0.2)'
        }}>
          <Typography variant="body2" sx={{ color: '#667eea', fontWeight: '600' }}>
            {groupNames.length === 0 
              ? `No pieces found matching "${searchQuery}"`
              : `Found ${Object.values(filteredGroupedPieces).flat().length} pieces matching "${searchQuery}" in ${groupNames.length} groups`
            }
          </Typography>
        </Box>
      )}

      {/* Groups Display */}
      {currentGroups.length > 0 ? (
        <>
          {currentGroups.map(groupName => 
            renderGroup(groupName, filteredGroupedPieces[groupName])
          )}
          
          {/* Main Groups Pagination */}
          {totalGroupPages > 1 && (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              mt: 4,
              p: 2,
              backgroundColor: 'rgba(102, 126, 234, 0.08)',
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
            {searchQuery ? 'No Results Found' : 'No Groups Found'}
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            {searchQuery 
              ? `No pieces match your search "${searchQuery}"`
              : 'No groups are available in the system'
            }
          </Typography>
          {searchQuery && (
            <Button 
              variant="outlined" 
              onClick={handleClearSearch}
              sx={{ mt: 2, color: '#667eea', borderColor: '#667eea' }}
            >
              Clear Search
            </Button>
          )}
        </EmptyState>
      )}
    </Container>
  );
}