from poisson_model import calculate_strengths, expected_ft_goals, predict_market
from get_helpers import load_data

# ============================================================
# SETTINGS
# ============================================================

# PREDICTION OPTIONS
# FT      = Exact full-time score
# HTFT    = Exact half-time / full-time score
# 1X2     = Home / Draw / Away
# BTTS    = Both teams to score
# OU25    = Over / Under 2.5 goals
# OU35    = Over / Under 3.5 goals
# HT_1X2  = Half-time Home / Draw / Away
# HT_OU15 = Half-time Over / Under 1.5 goals

PREDICTION = "BTTS"

fixtures = [
    ("Everton", "Chelsea"),
    ("Brentford", "Liverpool"),
    ("Fulham", "Hull"),
    ("Man City", "Ipswich"),
    ("Newcastle", "Aston Villa"),
]

# ============================================================
# RUN
# ============================================================

matches = load_data()
league = "EPL"
model = calculate_strengths(matches, league)

print("=" * 100)
print(f"FOOTBALL POISSON PREDICTIONS | {PREDICTION}")
print("=" * 100)
print(f"Model date: {model['ReferenceDate'].date()}")
print(f"Data: {matches['Date'].min().date()} to {matches['Date'].max().date()}")
print("=" * 100)

for home_team, away_team in fixtures:
    print(f"\n{home_team} vs {away_team}")

    if home_team not in model["Teams"] or away_team not in model["Teams"]:
        missing = [team for team in [home_team, away_team] if team not in model["Teams"]]
        print(f"SKIPPED - {', '.join(missing)} not available in model")
        continue

    lambda_home, lambda_away = expected_ft_goals(model, home_team, away_team)
    predictions = predict_market(model, home_team, away_team, PREDICTION)
    prediction = predictions.iloc[0]

    print(f"xG: {lambda_home:.2f} - {lambda_away:.2f}")

    if PREDICTION == "FT":
        print(f"PICK: {prediction['Score']} | Probability: {prediction['Probability_%']:.2f}% | Fair Odds: {prediction['Fair_Odds']:.2f}")
    elif PREDICTION == "HTFT":
        print(f"PICK: {prediction['HTFT_Score']} | Probability: {prediction['Probability_%']:.2f}% | Fair Odds: {prediction['Fair_Odds']:.2f}")
    else:
        print(f"PICK: {prediction['Outcome']} | Probability: {prediction['Probability_%']:.2f}% | Fair Odds: {prediction['Fair_Odds']:.2f}")