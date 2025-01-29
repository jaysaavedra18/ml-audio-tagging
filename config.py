# config.py

import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get today's date
today_date = datetime.datetime.now(tz=datetime.timezone.utc).date()
DATE_STRING = today_date.strftime("%Y-%m-%d")

# Define directory paths
BASE_DIRECTORY = Path(__file__).resolve().parent.parent
AUDIO_DIRECTORY = os.getenv("AUDIO_DIRECTORY")  # Set environment variable in .env file
DATA_DIRECTORY = Path(BASE_DIRECTORY) / "src" / "data"
DAILY_PLAYLIST_DIRECTORY = Path(AUDIO_DIRECTORY) / "output" / f"{DATE_STRING}"
DOWNLOADS_DIRECTORY = Path(AUDIO_DIRECTORY) / "misc"
LIBRARY_DIRECTORY = Path(AUDIO_DIRECTORY) / "lofi"

# Define data file paths
LIBRARY_DATA_PATH = Path(DATA_DIRECTORY) / "library_data.json"

# Define FMA dataset file paths
FMA_DATA_DIRECTORY = Path(AUDIO_DIRECTORY) / "fma_medium"
FMA_METADATA_DIRECTORY = Path(AUDIO_DIRECTORY) / "fma_metadata"
