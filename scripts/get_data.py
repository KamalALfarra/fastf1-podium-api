from pathlib import Path
import requests
import pandas as pd
import time


# ==========================
# Configuration
# ==========================

START_YEAR = 2014
END_YEAR = 2024

BASE_URL = "https://api.jolpi.ca/ergast/f1"

ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = ROOT / "data" / "raw"

RAW_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================
# Helpers
# ==========================

def save_csv(data, filename):
    path = RAW_FOLDER / filename

    if path.exists():
        print(f"{filename} already exists. Skipping.")
        return

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

    print(f"Saved {filename} ({len(df)} rows)")


def get_json(url, retries=3):
    """Fetches JSON from a URL with timeout and retry logic."""
    for attempt in range(retries):
        try:
            # A 15-second timeout prevents the script from hanging forever
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ReadTimeout:
            print(f"    [Timeout] API is slow. Retrying ({attempt + 1}/{retries})...")
            time.sleep(3)  # Give the server a breather before retrying
        except requests.exceptions.RequestException as e:
            print(f"    [Error] Request failed: {e}")
            break

    return None

# ==========================
# Races
# ==========================

def download_races():

    filename = "races.csv"

    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):

        print(f"Downloading races {year}")

        url = f"{BASE_URL}/{year}.json"

        data = get_json(url)

        if not data:
            continue

        races = (
            data["MRData"]
            ["RaceTable"]
            ["Races"]
        )

        for race in races:

            circuit = race["Circuit"]

            rows.append({

                "season": year,
                "round": race["round"],
                "raceName": race["raceName"],
                "circuitId": circuit["circuitId"],
                "circuitName": circuit["circuitName"],
                "locality": circuit["Location"]["locality"],
                "country": circuit["Location"]["country"],
                "date": race["date"]

            })

        time.sleep(0.2)


    save_csv(rows, filename)



# ==========================
# Race Results
# ==========================

def download_results():

    filename = "race_results.csv"

    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):

        print(f"Downloading results {year}")

        for podium_position in [1, 2, 3]:

            url = f"{BASE_URL}/{year}/results/{podium_position}.json"

            data = get_json(url)

            if not data:
                continue

            races = (
                data["MRData"]
                ["RaceTable"]
                ["Races"]
            )

            for race in races:

                race_name = race["raceName"]

                results = race["Results"]

                for result in results:

                    rows.append({

                        "season": year,
                        "round": race["round"],
                        "raceName": race_name,

                        "driverId":
                        result["Driver"]["driverId"],

                        "driver":
                        result["Driver"]["givenName"]
                        + " "
                        +
                        result["Driver"]["familyName"],

                        "constructor":
                        result["Constructor"]["name"],

                        "position":
                        result["position"],

                        "points":
                        result["points"]

                    })

            time.sleep(0.2)

    save_csv(rows, filename)

# ==========================
# Driver Standings
# ==========================
# Assuming RAW_FOLDER is defined earlier in your script, e.g., RAW_FOLDER = Path("data")

import time
import requests
import pandas as pd
from datetime import datetime


# Assuming RAW_FOLDER is defined earlier in your script, e.g., RAW_FOLDER = Path("data")

def fetch_ml_standings_data(start_year, end_year, filename="driver_ml_features.csv"):
    """
    Downloads driver standings per round alongside pre-race career features (age, experience, wins, podiums)
    and includes strictly PRE-RACE constructor/driver standings and grid positions to prevent ML data leakage.
    Handles API pagination correctly and strictly throttles requests to prevent IP bans.
    """
    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def get_json(url):
        # STRATEGY: Proactive delay. Wait 1.2 seconds BEFORE every single request.
        # This guarantees we never exceed ~50 requests per minute.
        time.sleep(1.2)

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    print("    [Rate Limit Hit] Cooling down for 10 seconds...")
                    time.sleep(10)  # Severe cooldown if we still somehow get blocked
            except requests.exceptions.RequestException:
                print("    [Network Error] Retrying in 5 seconds...")
                time.sleep(5)
        return None

    # --- Step 1: Fetch exact dates of birth for all drivers ---
    print("Step 1/4: Fetching driver metadata...")
    driver_dob = {}
    offset = 0
    while True:
        data = get_json(f"{BASE_URL}/drivers.json?limit=100&offset={offset}")
        if not data: break
        drivers = data.get("MRData", {}).get("DriverTable", {}).get("Drivers", [])
        if not drivers: break
        for d in drivers:
            driver_dob[d["driverId"]] = d.get("dateOfBirth")

        total = int(data.get("MRData", {}).get("total", 0))
        offset += 100
        if offset >= total:
            break

    # --- Step 2: Build In-Memory Career History ---
    # Tip: Set the start of this loop to 2014 instead of 1950 if you only want modern data
    history_start_year = 1990
    print(
        f"Step 2/4: Building career history from {history_start_year} to {end_year} to track past wins/podiums and grid positions...")
    driver_history = {}
    driver_debut = {}
    race_dates = {}

    for y in range(history_start_year, end_year + 1):
        print(f"  Fetching history for {y}...")
        offset = 0
        while True:
            data = get_json(f"{BASE_URL}/{y}/results.json?limit=100&offset={offset}")
            if not data: break

            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races: break

            for r in races:
                rnd = int(r["round"])
                race_dates[(y, rnd)] = r.get("date")

                for res in r.get("Results", []):
                    d_id = res.get("Driver", {}).get("driverId")
                    pos = res.get("position")
                    grid_pos = res.get("grid", "0")

                    if d_id not in driver_debut:
                        driver_debut[d_id] = y
                    if d_id not in driver_history:
                        driver_history[d_id] = []

                    driver_history[d_id].append({
                        "year": y,
                        "round": rnd,
                        "is_win": 1 if pos == "1" else 0,
                        "is_podium": 1 if pos in ["1", "2", "3"] else 0,
                        "grid": int(grid_pos) if grid_pos.isdigit() else 0
                    })

            total = int(data.get("MRData", {}).get("total", 0))
            offset += 100
            if offset >= total:
                break

    # --- Step 3: Fetch Standings for Target ML Years and Combine ---
    print(f"Step 3/4: Fetching PRE-RACE standings and calculating features ({start_year}-{end_year})...")
    ml_rows = []

    for y in range(start_year, end_year + 1):
        rounds_this_year = [k[1] for k in race_dates.keys() if k[0] == y]
        if not rounds_this_year: continue
        max_round = max(rounds_this_year)

        for rnd in range(1, max_round + 1):
            print(f"  Processing Season {y} - Round {rnd}...")

            prev_constructor_standings = {}
            prev_driver_standings = {}
            prev_driver_points = {}

            if rnd > 1:
                # Constructor Standings Before Race
                c_data_prev = get_json(f"{BASE_URL}/{y}/{rnd - 1}/constructorstandings.json?limit=100")
                if c_data_prev:
                    c_lists = c_data_prev.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
                    if c_lists:
                        for c_item in c_lists[0].get("ConstructorStandings", []):
                            c_id = c_item.get("Constructor", {}).get("constructorId")
                            if c_id:
                                prev_constructor_standings[c_id] = int(c_item.get("position", 0))

                # Driver Standings Before Race
                d_data_prev = get_json(f"{BASE_URL}/{y}/{rnd - 1}/driverstandings.json?limit=100")
                if d_data_prev:
                    d_lists = d_data_prev.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
                    if d_lists:
                        for d_item in d_lists[0].get("DriverStandings", []):
                            d_id_prev = d_item.get("Driver", {}).get("driverId")
                            if d_id_prev:
                                prev_driver_standings[d_id_prev] = int(d_item.get("position", 0))
                                prev_driver_points[d_id_prev] = float(d_item.get("points", 0))

            # Fetch current round list of drivers participating
            data = get_json(f"{BASE_URL}/{y}/{rnd}/driverstandings.json?limit=100")
            if not data: continue

            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists: continue

            drivers = standings_lists[0].get("DriverStandings", [])
            race_date_str = race_dates.get((y, rnd))

            for item in drivers:
                driver_info = item.get("Driver", {})
                d_id = driver_info.get("driverId")

                constructors_list = item.get("Constructors", [])
                c_name = ""
                c_standing_before_race = 0
                if constructors_list:
                    c_info = constructors_list[0]
                    c_name = c_info.get("name", "")
                    c_id = c_info.get("constructorId")
                    c_standing_before_race = prev_constructor_standings.get(c_id, 0)

                d_standing_before_race = prev_driver_standings.get(d_id, 0)
                d_points_before_race = prev_driver_points.get(d_id, 0.0)

                age = None
                dob_str = driver_dob.get(d_id) or driver_info.get("dateOfBirth")
                if race_date_str and dob_str:
                    try:
                        r_date = datetime.strptime(race_date_str, "%Y-%m-%d")
                        d_date = datetime.strptime(dob_str, "%Y-%m-%d")
                        age = round((r_date - d_date).days / 365.25, 2)
                    except ValueError:
                        pass

                experience = y - driver_debut.get(d_id, y)

                wins_before = 0
                podiums_before = 0
                grid_position = None

                for hist in driver_history.get(d_id, []):
                    if hist["year"] < y or (hist["year"] == y and hist["round"] < rnd):
                        wins_before += hist["is_win"]
                        podiums_before += hist["is_podium"]
                    elif hist["year"] == y and hist["round"] == rnd:
                        grid_position = hist["grid"]

                ml_rows.append({
                    "season": y,
                    "round": rnd,
                    "driver_name": f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                    "constructor_name": c_name,
                    "constructor_championship_standing": c_standing_before_race,
                    "championship_standing": d_standing_before_race,
                    "championship_points": d_points_before_race,
                    "grid_position": grid_position,
                    "age_at_race": age,
                    "years_of_experience": experience,
                    "career_wins_before_race": wins_before,
                    "career_podiums_before_race": podiums_before
                })

    # --- Step 4: Save everything to a single CSV ---
    print(f"Step 4/4: Saving {len(ml_rows)} rows to {filename}...")
    df = pd.DataFrame(ml_rows)
    path = RAW_FOLDER / filename
    df.to_csv(path, index=False)

    print("Done! Data is fully processed for Machine Learning.")
# To run it in your script (adjust the years to whatever you need):
# fetch_ml_standings_data(start_year=2010, end_year=2023)
# ==========================
# Merge Tables
# ==========================
def make_Final_table():
    # 1. Load the CSV files into DataFrames
    if (ROOT / "data" / "processed" / "final_features.csv").exists():
        print("data/processed/final_features.csv already exists. Skipping.")
        return
    # Replace 'all_drivers.csv' and 'podiums.csv' with your actual file paths
    df_all = pd.read_csv(ROOT / "data/raw/driver_ml_features.csv")
    df_podiums = pd.read_csv(ROOT / 'data/raw/race_results.csv')

    # 2. Rename 'driver' to 'driver_name' in the podiums DataFrame so the columns match
    df_podiums = df_podiums.rename(columns={'driver': 'driver_name'})

    # 3. Create our boolean indicator column in the podiums DataFrame
    df_podiums['got_podium'] = True

    # 4. Isolate only the columns we need for the merge so we don't duplicate other data
    # (like constructor names or points which exist in both tables)
    podium_keys = df_podiums[['season', 'round', 'driver_name', 'got_podium']]

    # 5. Perform a left join
    # This keeps everything in df_all and adds the 'got_podium' value where it finds a match
    df_merged = pd.merge(df_all, podium_keys, on=['season', 'round', 'driver_name'], how='left')

    # 6. Fill the NaN values with False for drivers who did not get a podium
    df_merged['got_podium'] = df_merged['got_podium'].fillna(False)

    # 7. Save the updated table to a new CSV file
    path = ROOT / "data" / "processed" / "final_features.csv"

    path.parent.mkdir(parents=True, exist_ok=True)

    df_merged.to_csv(path, index=False)
    print("File successfully saved as 'final_features.cvs'")

# ==========================
# Main
# ==========================

if __name__ == "__main__":


    print("Starting F1 dataset download...")


    download_races()

    download_results()

    fetch_ml_standings_data(start_year=START_YEAR,end_year=END_YEAR)
    make_Final_table()
    print("\nFinished.")

