from enum import Enum 

class ActionType(Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    VIEW_PIECE = "view_piece"
    ANNOTATE_PIECE = "annotate_piece"
    UPDATE_PROFILE = "update_profile"
    CREATE_PROFILE = "create_profile"