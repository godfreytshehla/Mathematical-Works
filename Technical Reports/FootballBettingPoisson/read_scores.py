import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "football_scores.csv"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "EPL": "E0",
    "LaLiga": "SP1",
}

SEASONS = {
    "2021": "2020-21",
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
    "2627": "2026-27",
}

# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_scores():
    all_matches = []

    print("=" * 70)
    print("DOWNLOADING FOOTBALL DATA")
    print("=" * 70)

    for league_name, league_code in LEAGUES.items():
        for season_code, season_name in SEASONS.items():
            url = f"{BASE_URL}/{season_code}/{league_code}.csv"
            print(f"{league_name:8} {season_name}: ", end="")

            try:
                df = pd.read_csv(url)
                required = ["Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "FTHG", "FTAG"]
                missing = [column for column in required if column not in df.columns]

                if missing:
                    print(f"SKIPPED - missing {missing}")
                    continue

                df = df[required].copy()
                df["League"] = league_name
                df["Season"] = season_name
                all_matches.append(df)

                print(f"OK ({len(df)} rows)")

            except Exception as error:
                print(f"FAILED ({error})")

    if not all_matches:
        raise RuntimeError("No football data could be downloaded.")

    return pd.concat(all_matches, ignore_index=True)

# ============================================================
# CLEAN DATA
# ============================================================

def clean_scores(matches):
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

    goal_columns = ["HTHG", "HTAG", "FTHG", "FTAG"]
    matches[goal_columns] = matches[goal_columns].apply(pd.to_numeric, errors="coerce")

    required = ["Date", "HomeTeam", "AwayTeam", "HTHG", "HTAG", "FTHG", "FTAG"]
    matches = matches.dropna(subset=required).copy()
    matches[goal_columns] = matches[goal_columns].astype(int)

    matches["SHHG"] = matches["FTHG"] - matches["HTHG"]
    matches["SHAG"] = matches["FTAG"] - matches["HTAG"]
    matches = matches[(matches["SHHG"] >= 0) & (matches["SHAG"] >= 0)].copy()

    matches["HT_Score"] = matches["HTHG"].astype(str) + "-" + matches["HTAG"].astype(str)
    matches["FT_Score"] = matches["FTHG"].astype(str) + "-" + matches["FTAG"].astype(str)
    matches["HTFT_Score"] = matches["HT_Score"] + " / " + matches["FT_Score"]

    matches = matches.sort_values(["Date", "League"]).reset_index(drop=True)

    columns = [
        "Date", "League", "Season", "HomeTeam", "AwayTeam",
        "HTHG", "HTAG", "HT_Score", "SHHG", "SHAG",
        "FTHG", "FTAG", "FT_Score", "HTFT_Score",
    ]

    return matches[columns]

# ============================================================
# SUMMARY
# ============================================================

def show_summary(matches):
    print()
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Matches: {len(matches):,}")
    print(f"First match: {matches['Date'].min().date()}")
    print(f"Latest match: {matches['Date'].max().date()}")

    print()
    print("Matches by league:")
    print(matches["League"].value_counts().to_string())

    print()
    print("Latest 10 matches:")

    columns = ["Date", "League", "HomeTeam", "AwayTeam", "HT_Score", "FT_Score", "HTFT_Score"]
    print(matches[columns].tail(10).to_string(index=False))

# ============================================================
# MAIN
# ============================================================

def main():
    matches = download_scores()
    matches = clean_scores(matches)
    matches.to_csv(OUTPUT_FILE, index=False)

    show_summary(matches)

    print()
    print("=" * 70)
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()