from __future__ import annotations

import math

import requests

from ..config import env

OWM = "https://api.openweathermap.org/data/2.5/weather"


def fetch_weather_for_coords(lat: float, lon: float) -> dict:
    key = env("OPENWEATHER_API_KEY")
    r = requests.get(
        OWM,
        params={"lat": lat, "lon": lon, "units": "imperial", "appid": key},
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    return {
        "temp_f": j["main"]["temp"],
        "wind_mph": j["wind"]["speed"],
        "wind_deg": j["wind"].get("deg", 0),
        "humidity": j["main"]["humidity"],
        "conditions": j["weather"][0]["main"],
        "precip_flag": 1 if j["weather"][0]["main"] in ("Rain", "Drizzle", "Thunderstorm", "Snow") else 0,
    }


def signed_wind(wind_mph: float, wind_deg: float, stadium_orientation_deg: float) -> float:
    """Positive if wind blowing toward center field (helpful), negative if blowing in."""
    delta = abs(((wind_deg - stadium_orientation_deg + 180) % 360) - 180)
    factor = math.cos(math.radians(delta))
    return wind_mph * factor


if __name__ == "__main__":
    w = fetch_weather_for_coords(42.3467, -71.0972)
    print(w)
