import pandas as pd
from pathlib import Path
from poisson_model import expected_ft_goals, expected_ht_goals, expected_sh_goals, ft_correct_scores, htft_correct_scores

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "football_scores.csv"
TOP_FT = 20
TOP_HTFT = 30

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE}. Run read_scores.py first.")

    matches = pd.read_csv(DATA_FILE)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches = matches.dropna(subset=["Date"]).copy()

    return matches

# ============================================================
# SELECT LEAGUE
# ============================================================

def select_league(matches):
    leagues = sorted(matches["League"].unique())

    print("\nLEAGUES")
    for number, league in enumerate(leagues, 1):
        print(f"{number}. {league}")

    while True:
        try:
            choice = int(input("\nSelect league: "))
            if 1 <= choice <= len(leagues):
                return leagues[choice - 1]
        except ValueError:
            pass

        print("Invalid selection.")

# ============================================================
# SELECT TEAM
# ============================================================

def select_team(teams, label):
    print(f"\n{label.upper()} TEAM")

    for number, team in enumerate(teams, 1):
        print(f"{number:2}. {team}")

    while True:
        try:
            choice = int(input(f"\nSelect {label.lower()} team: "))
            if 1 <= choice <= len(teams):
                return teams[choice - 1]
        except ValueError:
            pass

        print("Invalid selection.")

# ============================================================
# DISPLAY EXPECTED GOALS
# ============================================================

def show_expected_goals(model, home_team, away_team):
    ft_home, ft_away = expected_ft_goals(model, home_team, away_team)
    ht_home, ht_away = expected_ht_goals(model, home_team, away_team)
    sh_home, sh_away = expected_sh_goals(model, home_team, away_team)

    print("\n" + "=" * 70)
    print("EXPECTED GOALS")
    print("=" * 70)
    print(f"{'Period':<15}{home_team:>20}{away_team:>20}")
    print(f"{'Half Time':<15}{ht_home:>20.3f}{ht_away:>20.3f}")
    print(f"{'Second Half':<15}{sh_home:>20.3f}{sh_away:>20.3f}")
    print(f"{'Full Time':<15}{ft_home:>20.3f}{ft_away:>20.3f}")

# ============================================================
# DISPLAY FT SCORES
# ============================================================

def show_ft_scores(model, home_team, away_team):
    scores = ft_correct_scores(model, home_team, away_team)
    display = scores[["Score", "Probability_%", "Fair_Odds"]].head(TOP_FT).copy()
    display["Probability_%"] = display["Probability_%"].round(3)
    display["Fair_Odds"] = display["Fair_Odds"].round(2)

    print("\n" + "=" * 70)
    print("FULL-TIME CORRECT SCORE")
    print("=" * 70)
    print(display.to_string(index=False))

# ============================================================
# DISPLAY HT / FT SCORES
# ============================================================

def show_htft_scores(model, home_team, away_team):
    scores = htft_correct_scores(model, home_team, away_team)
    display = scores[["HT_Score", "FT_Score", "Probability_%", "Fair_Odds"]].head(TOP_HTFT).copy()
    display["Probability_%"] = display["Probability_%"].round(4)
    display["Fair_Odds"] = display["Fair_Odds"].round(2)

    print("\n" + "=" * 70)
    print("HALF-TIME / FULL-TIME EXACT CORRECT SCORE")
    print("=" * 70)
    print(display.to_string(index=False))


# ============================================================
# SELECT PREDICTION
# ============================================================

def select_prediction():
    print("\nPREDICT")
    print("1. FT Correct Score")
    print("2. HT/FT Correct Score")
    print("3. Both")

    while True:
        choice = input("\nSelect prediction: ").strip()
        if choice == "1":
            return "FT"
        if choice == "2":
            return "HTFT"
        if choice == "3":
            return "BOTH"

        print("Invalid selection.")