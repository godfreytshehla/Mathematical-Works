import requests

# ============================================================
# SETTINGS
# ============================================================

API_KEY = "Paste you api key" #get your own API key from https://the-odds-api.com/
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
}

# ============================================================
# EVENTS
# ============================================================

def get_events(league):
    sport = SPORTS[league]
    url = f"{BASE_URL}/sports/{sport}/events"
    params = {"apiKey": API_KEY}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

# ============================================================
# CORRECT SCORE ODDS
# ============================================================

def get_correct_score_odds(league, event_id):
    sport = SPORTS[league]
    url = f"{BASE_URL}/sports/{sport}/events/{event_id}/odds"
    params = {"apiKey": API_KEY, "regions": "eu", "markets": "correct_score", "oddsFormat": "decimal"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

# ============================================================
# FIND FIXTURE
# ============================================================

def find_event(league, home_team, away_team):
    events = get_events(league)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        if home_team.lower() in home.lower() and away_team.lower() in away.lower():
            return event

    return None

# ============================================================
# EXTRACT ODDS
# ============================================================

def extract_correct_score_odds(data):
    rows = []

    for bookmaker in data.get("bookmakers", []):
        bookmaker_name = bookmaker.get("title", "")

        for market in bookmaker.get("markets", []):
            if market.get("key") != "correct_score":
                continue

            for outcome in market.get("outcomes", []):
                score = outcome.get("name")
                odds = outcome.get("price")

                if score is not None and odds is not None:
                    rows.append({"Bookmaker": bookmaker_name, "Score": score, "Odds": float(odds)})

    return rows

# ============================================================
# BEST ODDS
# ============================================================

def get_best_correct_score_odds(league, home_team, away_team):
    event = find_event(league, home_team, away_team)

    if event is None:
        return {}

    data = get_correct_score_odds(league, event["id"])
    rows = extract_correct_score_odds(data)
    best = {}

    for row in rows:
        score = row["Score"]

        if score not in best or row["Odds"] > best[score]["Odds"]:
            best[score] = {"Odds": row["Odds"], "Bookmaker": row["Bookmaker"]}

    return best