from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


RAW_ROOT = "https://raw.githubusercontent.com/hudl/open-data/master/data"


def download_json(url: str):
    request = Request(url, headers={"User-Agent": "zcpv-research/0.1"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an official StatsBomb Open Data competition.")
    parser.add_argument("--competition", type=int, default=43, help="StatsBomb competition id")
    parser.add_argument("--season", type=int, default=106, help="StatsBomb season id")
    parser.add_argument("--output", type=Path, default=Path("data/raw/statsbomb"))
    args = parser.parse_args()

    competition_url = f"{RAW_ROOT}/matches/{args.competition}/{args.season}.json"
    matches = download_json(competition_url)
    write_json(args.output / "matches.json", matches)

    for number, match in enumerate(matches, start=1):
        match_id = match["match_id"]
        destination = args.output / "events" / f"{match_id}.json"
        if not destination.exists():
            write_json(destination, download_json(f"{RAW_ROOT}/events/{match_id}.json"))
        print(f"[{number:02d}/{len(matches):02d}] {match_id} {match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}")

    manifest = {
        "source": "StatsBomb Open Data",
        "repository": "https://github.com/hudl/open-data",
        "competition_id": args.competition,
        "season_id": args.season,
        "matches": len(matches),
        "attribution_required": True,
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Downloaded {len(matches)} matches to {args.output}")


if __name__ == "__main__":
    main()
