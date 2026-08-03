import os
import fastf1
from fastf1.ergast import Ergast
import pandas as pd
from pathlib import Path

# 1. Configuration
START_YEAR = 2002
END_YEAR = 2024
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the data directory and output CSV relative to project root
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "historical_race_results.csv"


def download_historical_data():
    """Downloads F1 race results from START_YEAR to END_YEAR."""

    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # Check if the data already exists locally
    if os.path.exists(OUTPUT_FILE):
        print(f"Data already exists at '{OUTPUT_FILE}'. Skipping download.")
        print("If you want to re-download, delete the file and run this script again.")
        return

    print(f"Downloading F1 data from {START_YEAR} to {END_YEAR}...")

    # Initialize the Ergast API client (FastF1's historical data tool)
    ergast = Ergast(result_type='pandas', auto_cast=True, limit=1000)
    all_seasons_data = []

    # Loop through each year
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            print(f"Fetching data for {year}...")
            # Query the race results for the entire season
            season_results = ergast.get_race_results(season=year)

            # The API returns a list of DataFrames (one per race)
            # We concatenate them into a single DataFrame for the year
            if season_results.content:
                year_df = pd.concat(season_results.content, ignore_index=True)
                # Add the season year as a column for easier filtering later
                year_df['season_year'] = year
                all_seasons_data.append(year_df)

        except Exception as e:
            print(f"Error fetching data for {year}: {e}")

    # Combine all years into one massive DataFrame
    if all_seasons_data:
        final_dataset = pd.concat(all_seasons_data, ignore_index=True)

        # Save to CSV
        final_dataset.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved {len(final_dataset)} rows to {OUTPUT_FILE}")
    else:
        print("Failed to download any data.")


if __name__ == "__main__":
    download_historical_data()