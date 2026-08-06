from pathlib import Path
import requests
import pandas as pd
import time


# ==========================
# Configuration
# ==========================

START_YEAR = 2002
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

        url = f"{BASE_URL}/{year}/results/3.json"

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
# Qualifying
# ==========================

def download_qualifying():

    filename = "qualifying.csv"

    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return


    rows = []


    for year in range(START_YEAR, END_YEAR + 1):

        print(f"Downloading qualifying {year}")


        url = f"{BASE_URL}/{year}/qualifying.json"

        data = get_json(url)


        if not data:
            continue


        races = (
            data["MRData"]
            ["RaceTable"]
            ["Races"]
        )


        for race in races:

            for result in race["QualifyingResults"]:

                rows.append({

                    "season": year,

                    "round":
                    race["round"],

                    "raceName":
                    race["raceName"],

                    "driver":
                    result["Driver"]["givenName"]
                    +
                    " "
                    +
                    result["Driver"]["familyName"],

                    "constructor":
                    result["Constructor"]["name"],

                    "position":
                    result["position"]

                })


        time.sleep(0.2)


    save_csv(rows, filename)



# ==========================
# Driver Standings
# ==========================

import time
import requests
import pandas as pd
from datetime import datetime

import time
import requests
import pandas as pd
from datetime import datetime


def fetch_ml_standings_data(start_year, end_year, filename="driver_ml_features.csv"):
    """
    Downloads driver standings per round alongside pre-race career features (age, experience, wins, podiums)
    without hitting API rate limits.
    """
    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def get_json(url):
        # Built-in retry mechanism to handle transient network issues without crashing
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    print("    [Rate Limit Hit] Cooling down for 5 seconds...")
                    time.sleep(5)
            except requests.exceptions.RequestException:
                time.sleep(3)
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
        offset += 100
        time.sleep(0.3)

    # --- Step 2: Build In-Memory Career History (1950 to end_year) ---
    # We load all race results once. This completely eliminates the 429 rate limit error.
    print(f"Step 2/4: Building career history from 1950 to {end_year} to track past wins/podiums...")
    driver_history = {}
    driver_debut = {}
    race_dates = {}

    for y in range(1950, end_year + 1):
        data = get_json(f"{BASE_URL}/{y}/results.json?limit=1000")
        if not data: continue
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for r in races:
            rnd = int(r["round"])
            race_dates[(y, rnd)] = r.get("date")

            for res in r.get("Results", []):
                d_id = res.get("Driver", {}).get("driverId")
                pos = res.get("position")

                if d_id not in driver_debut:
                    driver_debut[d_id] = y
                if d_id not in driver_history:
                    driver_history[d_id] = []

                driver_history[d_id].append({
                    "year": y,
                    "round": rnd,
                    "is_win": 1 if pos == "1" else 0,
                    "is_podium": 1 if pos in ["1", "2", "3"] else 0
                })
        time.sleep(0.3)  # Small delay to respect API limits

    # --- Step 3: Fetch Standings for Target ML Years and Combine ---
    print(f"Step 3/4: Fetching per-race standings and calculating pre-race features ({start_year}-{end_year})...")
    ml_rows = []

    for y in range(start_year, end_year + 1):
        # Determine how many rounds are in the current year
        rounds_this_year = [k[1] for k in race_dates.keys() if k[0] == y]
        if not rounds_this_year: continue
        max_round = max(rounds_this_year)

        for rnd in range(1, max_round + 1):
            data = get_json(f"{BASE_URL}/{y}/{rnd}/driverstandings.json")
            if not data: continue

            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists: continue

            drivers = standings_lists[0].get("DriverStandings", [])
            race_date_str = race_dates.get((y, rnd))

            for item in drivers:
                driver_info = item.get("Driver", {})
                d_id = driver_info.get("driverId")

                # Feature: Exact Age at the time of the race
                age = None
                dob_str = driver_dob.get(d_id) or driver_info.get("dateOfBirth")
                if race_date_str and dob_str:
                    try:
                        r_date = datetime.strptime(race_date_str, "%Y-%m-%d")
                        d_date = datetime.strptime(dob_str, "%Y-%m-%d")
                        age = round((r_date - d_date).days / 365.25, 2)
                    except ValueError:
                        pass

                # Feature: Years of Experience up to this year
                experience = y - driver_debut.get(d_id, y)

                # Features: Wins & Podiums (STRICTLY BEFORE this round to prevent ML data leakage)
                wins_before = 0
                podiums_before = 0
                for hist in driver_history.get(d_id, []):
                    # Only tally results if they happened in a previous year OR an earlier round this year
                    if hist["year"] < y or (hist["year"] == y and hist["round"] < rnd):
                        wins_before += hist["is_win"]
                        podiums_before += hist["is_podium"]

                # Compile the unified row
                ml_rows.append({
                    "season": y,
                    "round": rnd,
                    "driver_name": f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                    "championship_standing": int(item.get("position", 0)),
                    "championship_points": float(item.get("points", 0)),
                    "age_at_race": age,
                    "years_of_experience": experience,
                    "career_wins_before_race": wins_before,
                    "career_podiums_before_race": podiums_before
                })

            time.sleep(0.3)

    # --- Step 4: Save everything to a single CSV ---
    print(f"Step 4/4: Saving {len(ml_rows)} rows to {filename}...")
    df = pd.DataFrame(ml_rows)
    path = RAW_FOLDER / filename
    df.to_csv(path, index=False)

    print("Done! Data is fully processed for Machine Learning.")


# To run it in your script (adjust the years to whatever you need):
# fetch_ml_standings_data(start_year=2010, end_year=2023)
# ==========================
# Constructor Standings
# ==========================

def download_constructor_standings():

    filename = "constructor_standings.csv"

    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):

        print(f"Downloading constructor standings {year}")

        url = f"{BASE_URL}/{year}/{round}/constructorStandings.json"

        data = get_json(url)

        if not data:
            continue

        standings = (
            data["MRData"]
            ["StandingsTable"]
            ["StandingsLists"]
        )

        if len(standings) == 0:
            continue


        constructors = standings[0].get(
            "ConstructorStandings",
            []
        )


        for item in constructors:

            constructor = item.get(
                "Constructor",
                {}
            )

            rows.append({

                "season": year,

                "position":
                    item.get("positionText"),

                "constructorId":
                    constructor.get("constructorId"),

                "constructor":
                    constructor.get("name"),

                "points":
                    item.get("points"),

                "wins":
                    item.get("wins")

            })


        time.sleep(0.2)


    save_csv(rows, filename)

# ==========================
# Main
# ==========================

if __name__ == "__main__":


    print("Starting F1 dataset download...")


    download_races()

    download_results()

    download_qualifying()

    fetch_ml_standings_data(start_year=START_YEAR,end_year=END_YEAR)

    download_constructor_standings()


    print("\nFinished.")

