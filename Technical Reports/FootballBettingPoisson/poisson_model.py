import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

MAX_GOALS = 8
HALF_LIFE_DAYS = 365
MIN_MATCHES = 5
EPSILON = 1e-10

# ============================================================
# POISSON
# ============================================================

def poisson_probabilities(expected_goals, max_goals=MAX_GOALS):
    expected_goals = float(expected_goals)
    probabilities = np.zeros(max_goals + 1, dtype=float)
    probabilities[0] = np.exp(-expected_goals)

    for goals in range(1, max_goals + 1):
        probabilities[goals] = probabilities[goals - 1] * expected_goals / goals

    return probabilities

def poisson_probability(goals, expected_goals):
    return poisson_probabilities(expected_goals, goals)[goals]

# ============================================================
# WEIGHTED CALCULATIONS
# ============================================================

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()

    if total_weight <= 0:
        return np.nan

    return np.sum(values * weights) / total_weight

def add_recency_weights(matches, reference_date, half_life_days=HALF_LIFE_DAYS):
    matches = matches.copy()
    reference_date = pd.Timestamp(reference_date)
    matches = matches[matches["Date"] <= reference_date].copy()
    age_days = (reference_date - matches["Date"]).dt.days.to_numpy()
    decay_rate = np.log(2) / half_life_days
    matches["Weight"] = np.exp(-decay_rate * age_days)

    return matches

# ============================================================
# TEAM STRENGTHS
# ============================================================

def calculate_strengths(matches, league, reference_date=None, half_life_days=HALF_LIFE_DAYS, min_matches=MIN_MATCHES):
    data = matches[matches["League"] == league].copy()

    if data.empty:
        raise ValueError(f"No data found for league: {league}")

    data["Date"] = pd.to_datetime(data["Date"])

    if reference_date is None:
        reference_date = data["Date"].max()

    reference_date = pd.Timestamp(reference_date)
    data = data[data["Date"] <= reference_date].copy()

    if data.empty:
        raise ValueError("No historical data available before the reference date.")

    data = add_recency_weights(data, reference_date, half_life_days)

    league_home_ft = weighted_mean(data["FTHG"], data["Weight"])
    league_away_ft = weighted_mean(data["FTAG"], data["Weight"])
    league_home_ht = weighted_mean(data["HTHG"], data["Weight"])
    league_away_ht = weighted_mean(data["HTAG"], data["Weight"])
    league_home_sh = weighted_mean(data["SHHG"], data["Weight"])
    league_away_sh = weighted_mean(data["SHAG"], data["Weight"])

    teams = np.union1d(data["HomeTeam"].unique(), data["AwayTeam"].unique())
    team_strengths = {}

    for team in teams:
        home = data[data["HomeTeam"] == team]
        away = data[data["AwayTeam"] == team]

        if len(home) < min_matches or len(away) < min_matches:
            continue

        home_ft_scored = weighted_mean(home["FTHG"], home["Weight"])
        home_ft_conceded = weighted_mean(home["FTAG"], home["Weight"])
        away_ft_scored = weighted_mean(away["FTAG"], away["Weight"])
        away_ft_conceded = weighted_mean(away["FTHG"], away["Weight"])

        home_attack_ft = home_ft_scored / np.maximum(league_home_ft, EPSILON)
        home_defence_ft = home_ft_conceded / np.maximum(league_away_ft, EPSILON)
        away_attack_ft = away_ft_scored / np.maximum(league_away_ft, EPSILON)
        away_defence_ft = away_ft_conceded / np.maximum(league_home_ft, EPSILON)

        home_ht_scored = weighted_mean(home["HTHG"], home["Weight"])
        home_ht_conceded = weighted_mean(home["HTAG"], home["Weight"])
        away_ht_scored = weighted_mean(away["HTAG"], away["Weight"])
        away_ht_conceded = weighted_mean(away["HTHG"], away["Weight"])

        home_attack_ht = home_ht_scored / np.maximum(league_home_ht, EPSILON)
        home_defence_ht = home_ht_conceded / np.maximum(league_away_ht, EPSILON)
        away_attack_ht = away_ht_scored / np.maximum(league_away_ht, EPSILON)
        away_defence_ht = away_ht_conceded / np.maximum(league_home_ht, EPSILON)

        home_sh_scored = weighted_mean(home["SHHG"], home["Weight"])
        home_sh_conceded = weighted_mean(home["SHAG"], home["Weight"])
        away_sh_scored = weighted_mean(away["SHAG"], away["Weight"])
        away_sh_conceded = weighted_mean(away["SHHG"], away["Weight"])

        home_attack_sh = home_sh_scored / np.maximum(league_home_sh, EPSILON)
        home_defence_sh = home_sh_conceded / np.maximum(league_away_sh, EPSILON)
        away_attack_sh = away_sh_scored / np.maximum(league_away_sh, EPSILON)
        away_defence_sh = away_sh_conceded / np.maximum(league_home_sh, EPSILON)

        team_strengths[team] = {
            "HomeAttackFT": home_attack_ft, "HomeDefenceFT": home_defence_ft,
            "AwayAttackFT": away_attack_ft, "AwayDefenceFT": away_defence_ft,
            "HomeAttackHT": home_attack_ht, "HomeDefenceHT": home_defence_ht,
            "AwayAttackHT": away_attack_ht, "AwayDefenceHT": away_defence_ht,
            "HomeAttackSH": home_attack_sh, "HomeDefenceSH": home_defence_sh,
            "AwayAttackSH": away_attack_sh, "AwayDefenceSH": away_defence_sh,
            "HomeMatches": len(home), "AwayMatches": len(away),
        }

    return {
        "League": league, "ReferenceDate": reference_date, "HalfLifeDays": half_life_days,
        "LeagueHomeFT": league_home_ft, "LeagueAwayFT": league_away_ft,
        "LeagueHomeHT": league_home_ht, "LeagueAwayHT": league_away_ht,
        "LeagueHomeSH": league_home_sh, "LeagueAwaySH": league_away_sh,
        "Teams": team_strengths,
    }

# ============================================================
# EXPECTED GOALS
# ============================================================

def validate_teams(model, home_team, away_team):
    if home_team not in model["Teams"]:
        raise ValueError(f"Unknown home team: {home_team}")
    if away_team not in model["Teams"]:
        raise ValueError(f"Unknown away team: {away_team}")

def expected_ft_goals(model, home_team, away_team):
    validate_teams(model, home_team, away_team)
    home = model["Teams"][home_team]
    away = model["Teams"][away_team]
    lambda_home = model["LeagueHomeFT"] * home["HomeAttackFT"] * away["AwayDefenceFT"]
    lambda_away = model["LeagueAwayFT"] * away["AwayAttackFT"] * home["HomeDefenceFT"]

    return lambda_home, lambda_away

def expected_ht_goals(model, home_team, away_team):
    validate_teams(model, home_team, away_team)
    home = model["Teams"][home_team]
    away = model["Teams"][away_team]
    lambda_home = model["LeagueHomeHT"] * home["HomeAttackHT"] * away["AwayDefenceHT"]
    lambda_away = model["LeagueAwayHT"] * away["AwayAttackHT"] * home["HomeDefenceHT"]

    return lambda_home, lambda_away

def expected_sh_goals(model, home_team, away_team):
    validate_teams(model, home_team, away_team)
    home = model["Teams"][home_team]
    away = model["Teams"][away_team]
    lambda_home = model["LeagueHomeSH"] * home["HomeAttackSH"] * away["AwayDefenceSH"]
    lambda_away = model["LeagueAwaySH"] * away["AwayAttackSH"] * home["HomeDefenceSH"]

    return lambda_home, lambda_away

# ============================================================
# SCORE MATRIX
# ============================================================

def score_matrix(lambda_home, lambda_away, max_goals=MAX_GOALS):
    home = poisson_probabilities(lambda_home, max_goals)
    away = poisson_probabilities(lambda_away, max_goals)
    matrix = np.outer(home, away)
    return matrix / matrix.sum()

# ============================================================
# FT CORRECT SCORE
# ============================================================

def ft_correct_scores(model, home_team, away_team, max_goals=MAX_GOALS):
    lambda_home, lambda_away = expected_ft_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away, max_goals)
    home_goals, away_goals = np.indices(matrix.shape)
    probabilities = matrix.ravel()
    results = pd.DataFrame({"HomeGoals": home_goals.ravel(), "AwayGoals": away_goals.ravel(), "Probability": probabilities})
    results["Score"] = results["HomeGoals"].astype(str) + "-" + results["AwayGoals"].astype(str)
    results["Probability_%"] = results["Probability"] * 100
    results["Fair_Odds"] = np.divide(1.0, results["Probability"], out=np.full(len(results), np.inf), where=results["Probability"] > 0)

    return results[["Score", "HomeGoals", "AwayGoals", "Probability", "Probability_%", "Fair_Odds"]].sort_values("Probability", ascending=False).reset_index(drop=True)

# ============================================================
# 1X2
# ============================================================

def match_result_probabilities(model, home_team, away_team):
    lambda_home, lambda_away = expected_ft_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away)
    probabilities = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
    outcomes = ["HOME", "DRAW", "AWAY"]

    return market_dataframe(outcomes, probabilities)

# ============================================================
# BOTH TEAMS TO SCORE
# ============================================================

def btts_probabilities(model, home_team, away_team):
    lambda_home, lambda_away = expected_ft_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away)
    no = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]
    probabilities = np.array([1 - no, no])

    return market_dataframe(["YES", "NO"], probabilities)

# ============================================================
# OVER / UNDER
# ============================================================

def over_under_probabilities(model, home_team, away_team, line=2.5):
    lambda_home, lambda_away = expected_ft_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away)
    home_goals, away_goals = np.indices(matrix.shape)
    totals = home_goals + away_goals
    over = matrix[totals > line].sum()
    under = matrix[totals < line].sum()

    return market_dataframe([f"OVER {line}", f"UNDER {line}"], np.array([over, under]))

# ============================================================
# HALF-TIME 1X2
# ============================================================

def ht_result_probabilities(model, home_team, away_team):
    lambda_home, lambda_away = expected_ht_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away)
    probabilities = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])

    return market_dataframe(["HOME", "DRAW", "AWAY"], probabilities)

# ============================================================
# HALF-TIME OVER / UNDER
# ============================================================

def ht_over_under_probabilities(model, home_team, away_team, line=1.5):
    lambda_home, lambda_away = expected_ht_goals(model, home_team, away_team)
    matrix = score_matrix(lambda_home, lambda_away)
    home_goals, away_goals = np.indices(matrix.shape)
    totals = home_goals + away_goals
    over = matrix[totals > line].sum()
    under = matrix[totals < line].sum()

    return market_dataframe([f"OVER {line}", f"UNDER {line}"], np.array([over, under]))

# ============================================================
# HT / FT EXACT CORRECT SCORE
# ============================================================

def htft_correct_scores(model, home_team, away_team, max_ht_goals=5, max_sh_goals=6):
    lambda_ht_home, lambda_ht_away = expected_ht_goals(model, home_team, away_team)
    lambda_sh_home, lambda_sh_away = expected_sh_goals(model, home_team, away_team)

    p_ht_home = poisson_probabilities(lambda_ht_home, max_ht_goals)
    p_ht_away = poisson_probabilities(lambda_ht_away, max_ht_goals)
    p_sh_home = poisson_probabilities(lambda_sh_home, max_sh_goals)
    p_sh_away = poisson_probabilities(lambda_sh_away, max_sh_goals)

    probabilities = p_ht_home[:, None, None, None] * p_ht_away[None, :, None, None] * p_sh_home[None, None, :, None] * p_sh_away[None, None, None, :]
    probabilities = probabilities / probabilities.sum()

    ht_home, ht_away, sh_home, sh_away = np.indices(probabilities.shape)
    probability = probabilities.ravel()
    ht_home = ht_home.ravel()
    ht_away = ht_away.ravel()
    ft_home = ht_home + sh_home.ravel()
    ft_away = ht_away + sh_away.ravel()

    results = pd.DataFrame({"HT_HomeGoals": ht_home, "HT_AwayGoals": ht_away, "FT_HomeGoals": ft_home, "FT_AwayGoals": ft_away, "Probability": probability})
    results["HT_Score"] = results["HT_HomeGoals"].astype(str) + "-" + results["HT_AwayGoals"].astype(str)
    results["FT_Score"] = results["FT_HomeGoals"].astype(str) + "-" + results["FT_AwayGoals"].astype(str)
    results["HTFT_Score"] = results["HT_Score"] + " / " + results["FT_Score"]
    results["Probability_%"] = results["Probability"] * 100
    results["Fair_Odds"] = np.divide(1.0, results["Probability"], out=np.full(len(results), np.inf), where=results["Probability"] > 0)

    return results[["HT_Score", "FT_Score", "HTFT_Score", "Probability", "Probability_%", "Fair_Odds"]].sort_values("Probability", ascending=False).reset_index(drop=True)

# ============================================================
# MARKET RESULTS
# ============================================================

def market_dataframe(outcomes, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    fair_odds = np.divide(1.0, probabilities, out=np.full(probabilities.shape, np.inf), where=probabilities > 0)

    return pd.DataFrame({"Outcome": outcomes, "Probability": probabilities, "Probability_%": probabilities * 100, "Fair_Odds": fair_odds}).sort_values("Probability", ascending=False).reset_index(drop=True)

# ============================================================
# PREDICT MARKET
# ============================================================

def predict_market(model, home_team, away_team, market):
    if market == "FT":
        return ft_correct_scores(model, home_team, away_team)
    if market == "HTFT":
        return htft_correct_scores(model, home_team, away_team)
    if market == "1X2":
        return match_result_probabilities(model, home_team, away_team)
    if market == "BTTS":
        return btts_probabilities(model, home_team, away_team)
    if market == "OU25":
        return over_under_probabilities(model, home_team, away_team, 2.5)
    if market == "OU35":
        return over_under_probabilities(model, home_team, away_team, 3.5)
    if market == "HT_1X2":
        return ht_result_probabilities(model, home_team, away_team)
    if market == "HT_OU15":
        return ht_over_under_probabilities(model, home_team, away_team, 1.5)

    raise ValueError(f"Unknown prediction market: {market}")