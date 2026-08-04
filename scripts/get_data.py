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


def get_json(url):
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed:", url)
        return None

    return response.json()


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

def download_driver_standings():

    filename = "driver_standings.csv"

    if (RAW_FOLDER / filename).exists():
        print(f"{filename} already exists. Skipping.")
        return

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):

        print(f"Downloading driver standings {year}")

        url = f"{BASE_URL}/{year}/driverstandings.json"

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


        drivers = standings[0].get("DriverStandings", [])


        for item in drivers:

            driver = item.get("Driver", {})

            rows.append({

                "season": year,

                "position": item.get("positionText"),

                "driverId": driver.get("driverId"),

                "driver":
                    driver.get("givenName", "")
                    + " "
                    +
                    driver.get("familyName", ""),

                "points": item.get("points"),

                "wins": item.get("wins")

            })


        time.sleep(0.2)


    save_csv(rows, filename)

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

        url = f"{BASE_URL}/{year}/constructorstandings.json"

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

    download_driver_standings()

    download_constructor_standings()


    print("\nFinished.")