import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data" / "processed" / "final_features.csv"
)

# Seasons to evaluate
TEST_SEASONS = [ 2022, 2023, 2024]

test_df = df[df["season"].isin(TEST_SEASONS)].copy()

# Remove rows without a valid grid position
test_df = test_df[test_df["grid_position"].notna()].copy()

test_df["grid_rank"] = (
    test_df
    .groupby(["season", "round"])["grid_position"]
    .rank(method="first", ascending=True)
)

# Top 3 on the grid = predicted podium
test_df["predicted_podium"] = test_df["grid_rank"] <= 3

print("\n--- Baseline: Top 3 Grid Positions ---")

all_correct = 0
all_races = 0

for season in TEST_SEASONS:

    season_df = test_df[test_df["season"] == season]

    correct = (
        season_df["predicted_podium"]
        & season_df["got_podium"]
    ).sum()

    races = (
        season_df[["season", "round"]]
        .drop_duplicates()
        .shape[0]
    )

    average_correct = correct / races
    recall_at_3 = average_correct / 3

    all_correct += correct
    all_races += races

    print(
        f"{season}: "
        f"{average_correct:.2f} / 3 correct | "
        f"Recall@3: {recall_at_3:.2%}"
    )


overall_average = all_correct / all_races
overall_recall_at_3 = overall_average / 3

print("\n--- Overall Baseline ---")
print(f"Seasons: {TEST_SEASONS}")
print(f"Total races: {all_races}")
print(f"Average correct podiums per race: {overall_average:.2f} / 3")
print(f"Recall@3: {overall_recall_at_3:.2%}")