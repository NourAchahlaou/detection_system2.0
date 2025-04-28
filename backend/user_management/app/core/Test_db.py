import sys
import os

try:
    # Add project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from settings import settings
    from sqlalchemy import create_engine

    engine = create_engine(settings.DATABASE_URI)
    connection = engine.connect()
    print("Connected successfully!")
    connection.close()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    input("Press Enter to exit...")
