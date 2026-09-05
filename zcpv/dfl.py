from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from math import atan2
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .actions import Action, value_action
from .pitch_control import PitchControlConfig, leave_one_out_zone_values
from .zones import ZoneGrid, fit_zone_values


@dataclass
class DFLPlayer:
    id: str
    name: str
    number: int
    role: str
    team_id: str
    team_name: str
    team_index: int


def read_metadata(path: Path) -> dict:
    root = ET.parse(path).getroot()
    info = root.find('MatchInformation')
    general = info.find('General')
    players: list[DFLPlayer] = []
    team_names = []
    home_id, away_id = general.get('HomeTeamId'), general.get('GuestTeamId')
    for team in info.find('Teams'):
        team_id, team_name = team.get('TeamId'), team.get('TeamName')
        team_index = 0 if team_id == home_id else 1
        team_names.append((team_index, team_name))
        for player in team.find('Players'):
            players.append(DFLPlayer(
                id=player.get('PersonId'), name=player.get('Shortname') or f"{player.get('FirstName','')} {player.get('LastName','')}".strip(),
                number=int(player.get('ShirtNumber') or 0), role=player.get('PlayingPosition') or 'SUB',
                team_id=team_id, team_name=team_name, team_index=team_index,
            ))
    team_names.sort()
    return {
        'match_id': general.get('MatchId').replace('DFL-MAT-', ''), 'title': general.get('MatchTitle'),
        'score': general.get('Result'), 'season': general.get('Season'), 'matchday': general.get('MatchDay'),
        'kickoff': general.get('PlannedKickoffTime'), 'teams': [x[1] for x in team_names], 'players': players,
    }


def _descendant(event: ET.Element, tag: str) -> ET.Element | None:
    return next((element for element in event.iter() if element.tag == tag), None)


def _event_rows(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    kickoffs = {}
    for event in root.findall('Event'):
        for node in event.iter():
            segment = node.get('GameSection')
            if segment in {'firstHalf', 'secondHalf'} and node.tag in {'KickOff', 'Kickoff'}:
                kickoffs[segment] = datetime.fromisoformat(event.get('EventTime'))
    rows = []
    for event in root.findall('Event'):
        play, shot, tackle = _descendant(event, 'Play'), _descendant(event, 'ShotAtGoal'), _descendant(event, 'TacklingGame')
        timestamp = datetime.fromisoformat(event.get('EventTime'))
        segment = 'secondHalf' if kickoffs.get('secondHalf') and timestamp >= kickoffs['secondHalf'] else 'firstHalf'
        actor = shot or play or tackle
        if actor is None:
            continue
        kind = 'shot' if shot is not None else ('tackle' if tackle is not None else 'pass' if _descendant(event, 'Pass') is not None else 'play')
        player = actor.get('Player') or actor.get('Winner')
        team = actor.get('Team') or actor.get('WinnerTeam')
        success = actor.get('Evaluation') in {'successfullyCompleted', 'successful'} or _descendant(event, 'SuccessfulShot') is not None
        try:
            x, y = float(event.get('X-Source-Position')), float(event.get('Y-Source-Position'))
            to_x, to_y = float(event.get('X-Position')), float(event.get('Y-Position'))
        except (TypeError, ValueError):
            continue
        base_minute = 45 if segment == 'secondHalf' else 0
        minute = base_minute + max(0, int((timestamp - kickoffs.get(segment, timestamp)).total_seconds() // 60))
        rows.append({'timestamp': timestamp, 'minute': minute, 'segment': segment, 'kind': kind, 'player': player, 'team': team, 'success': success, 'goal': _descendant(event, 'SuccessfulShot') is not None, 'x': x, 'y': y, 'to_x': to_x, 'to_y': to_y})
    return rows


def _attacking_directions(rows: list[dict]) -> dict[tuple[str, str], bool]:
    shots = defaultdict(list)
    for row in rows:
        if row['kind'] == 'shot' and row['team']:
            shots[(row['team'], row['segment'])].append(row['x'])
    teams = sorted({r['team'] for r in rows if r['team']})
    directions = {}
    for segment in ('firstHalf', 'secondHalf'):
        for team in teams:
            values = shots.get((team, segment), [])
            if values:
                directions[(team, segment)] = float(np.median(values)) > 52.5
        if len(teams) == 2:
            if (teams[0], segment) in directions and (teams[1], segment) not in directions:
                directions[(teams[1], segment)] = not directions[(teams[0], segment)]
            elif (teams[1], segment) in directions and (teams[0], segment) not in directions:
                directions[(teams[0], segment)] = not directions[(teams[1], segment)]
    for team in teams:
        directions.setdefault((team, 'firstHalf'), True)
        directions.setdefault((team, 'secondHalf'), not directions[(team, 'firstHalf')])
    return directions


def transformed_events(path: Path) -> list[dict]:
    rows = _event_rows(path)
    directions = _attacking_directions(rows)
    for row in rows:
        positive = directions.get((row['team'], row['segment']), True)
        if not positive:
            row['x'], row['to_x'] = 105.0 - row['x'], 105.0 - row['to_x']
            row['y'], row['to_y'] = 68.0 - row['y'], 68.0 - row['to_y']
    return rows


def learn_zone_values(event_paths: list[Path]) -> np.ndarray:
    grid = ZoneGrid(); transitions = np.zeros((12, 12)); shots = np.zeros(12); goals = np.zeros(12); turnovers = np.zeros(12)
    for path in event_paths:
        for row in transformed_events(path):
            origin = int(grid.index(row['x'], row['y']))
            if row['kind'] == 'shot':
                shots[origin] += 1; goals[origin] += int(row['goal'])
            elif row['kind'] == 'pass' and row['success']:
                destination = int(grid.index(row['to_x'], row['to_y'])); transitions[origin, destination] += 1
            elif row['kind'] == 'pass':
                turnovers[origin] += 1
    # Weak empirical prior avoids zero-value zones in a seven-match sample.
    shots += np.linspace(1, 4, 12); goals += np.linspace(.01, .35, 12)
    return fit_zone_values(transitions, shots, goals, turnovers=turnovers)


def value_match_actions(path: Path, players: list[DFLPlayer], zone_values: np.ndarray) -> tuple[dict[str, float], list[dict]]:
    grid = ZoneGrid(); totals = defaultdict(float); details = []; player_ids = {p.id for p in players}
    for row in transformed_events(path):
        if row['player'] not in player_ids:
            continue
        origin, destination = int(grid.index(row['x'], row['y'])), int(grid.index(row['to_x'], row['to_y']))
        try:
            if row['kind'] == 'shot':
                dx, dy = max(.1, 105-row['x']), abs(34-row['y']); angle = abs(atan2(7.32*dx, dx*dx+dy*dy-(7.32/2)**2))
                action = Action(row['player'], 'shot', origin, distance_m=(dx*dx+dy*dy)**.5, angle_rad=angle)
            elif row['kind'] == 'pass':
                action = Action(row['player'], 'pass', origin, destination, row['success'])
            elif row['kind'] == 'tackle':
                action = Action(row['player'], 'tackle', origin, successful=row['success'])
            else:
                continue
            value = value_action(action, zone_values); totals[row['player']] += value
            details.append({'player': row['player'], 'type': row['kind'].title(), 'value': round(value, 4), 'minute': row['minute'], 'from': origin, 'to': destination, 'detail': 'Successful' if row['success'] else 'Unsuccessful'})
        except ValueError:
            continue
    return totals, details


def read_tracking(path: Path, metadata: dict, target_hz: int = 5):
    frames_by_segment: dict[str, set[int]] = defaultdict(set); live_keys = set(); possession_by_key = {}; stride = None; source_hz = 25.0
    for _, elem in ET.iterparse(path, events=('end',)):
        if elem.tag != 'FrameSet' or (elem.get('TeamId') or '').lower() != 'ball':
            continue
        frames = list(elem)
        if len(frames) > 1:
            t0, t1 = datetime.fromisoformat(frames[0].get('T')), datetime.fromisoformat(frames[1].get('T'))
            source_hz = round(1 / (t1-t0).total_seconds()); stride = max(1, round(source_hz / target_hz))
        stride = stride or 5; segment = elem.get('GameSection')
        for frame in frames:
            n = int(frame.get('N'))
            if n % stride == 0:
                key = (segment, n); frames_by_segment[segment].add(n)
                if frame.get('BallStatus') == '1': live_keys.add(key)
                possession_by_key[key] = int(float(frame.get('BallPossession') or 0))
        elem.clear()
    keys = [(segment, n) for segment in ('firstHalf','secondHalf') for n in sorted(frames_by_segment[segment])]
    frame_index = {key:i for i,key in enumerate(keys)}; players = metadata['players']; player_index = {p.id:i for i,p in enumerate(players)}
    positions = np.full((len(keys), len(players), 2), np.nan, dtype=np.float32)
    for _, elem in ET.iterparse(path, events=('end',)):
        if elem.tag != 'FrameSet': continue
        pid, segment = elem.get('PersonId'), elem.get('GameSection')
        if pid in player_index:
            pidx = player_index[pid]
            for frame in elem:
                idx = frame_index.get((segment, int(frame.get('N'))))
                if idx is not None: positions[idx,pidx] = (float(frame.get('X'))+52.5, float(frame.get('Y'))+34.0)
        elem.clear()
    velocities = np.zeros_like(positions); dt = 1.0 / target_hz
    both = np.isfinite(positions[1:]).all(-1) & np.isfinite(positions[:-1]).all(-1)
    diff = (positions[1:] - positions[:-1]) / dt; velocities[1:][both] = diff[both]
    teams = np.array([p.team_index for p in players], dtype=np.int8)
    active = np.isfinite(positions).all(-1)
    live = np.array([key in live_keys for key in keys], dtype=bool)
    possessions = np.array([possession_by_key.get(key, 0) for key in keys], dtype=np.int8)
    segments = np.array([key[0] for key in keys])
    frame_numbers = np.array([key[1] for key in keys])
    return positions, velocities, teams, active, live, possessions, segments, frame_numbers, source_hz


def spatial_match(path: Path, event_path: Path, metadata: dict, zone_values: np.ndarray, target_hz: int = 5, chunk_size: int = 240) -> tuple[np.ndarray, np.ndarray, dict]:
    positions, velocities, teams, active, live, possessions, segments, frame_numbers, source_hz = read_tracking(path, metadata, target_hz)
    players = metadata['players']; totals = np.zeros((len(players),12), dtype=np.float64); cfg = PitchControlConfig(grid_x=32, grid_y=21)
    direction = _attacking_directions(_event_rows(event_path)); team_ids = {p.team_index:p.team_id for p in players}
    for segment in ('firstHalf','secondHalf'):
        chosen = np.flatnonzero((segments == segment) & live)
        if not len(chosen): continue
        values_by_team = np.zeros((2,12),dtype=np.float32); relative_zone = np.zeros((2,12),dtype=np.int8)
        for team in (0,1):
            positive = direction.get((team_ids[team],segment), True)
            for absolute in range(12):
                xband,yband=divmod(absolute,3); relative=absolute if positive else (3-xband)*3+(2-yband)
                relative_zone[team,absolute]=relative; values_by_team[team,absolute]=zone_values[relative]
        spatial_values=np.zeros((len(chosen),len(players),12),dtype=np.float32)
        for start in range(0,len(chosen),chunk_size):
            idx=chosen[start:start+chunk_size]
            spatial_values[start:start+len(idx)] = leave_one_out_zone_values(positions[idx],velocities[idx],teams,values_by_team,cfg)
        # Average within each possession so long possessions do not get extra weight merely from duration.
        possession_ids=[]; current=-1; previous_frame=None; previous_team=None
        for local,idx in enumerate(chosen):
            team=possessions[idx]
            if team not in (1,2) or previous_team!=team or previous_frame is None or frame_numbers[idx]-previous_frame > round(source_hz*2): current+=1
            possession_ids.append(current); previous_team=team; previous_frame=frame_numbers[idx]
        for possession_id in np.unique(possession_ids):
            mask=np.asarray(possession_ids)==possession_id
            for p in range(len(players)):
                p_active=active[chosen[mask],p]
                if not p_active.any(): continue
                absolute_mean=spatial_values[mask,p][p_active].mean(axis=0)
                for absolute,value in enumerate(absolute_mean): totals[p,relative_zone[teams[p],absolute]] += value
    active_frames = active.sum(axis=0); minutes = active_frames / target_hz / 60.0
    per90_zones = totals * 90 / np.maximum(minutes[:,None],1)
    expected_slots=np.minimum(active.sum(axis=1),22).sum()
    quality = {'source_hz': source_hz, 'analysis_hz': target_hz, 'sampled_frames': len(positions), 'live_frames': int(live.sum()), 'coverage': float(expected_slots / max(1, len(positions)*22))}
    return per90_zones, minutes, quality


def crunch_match(match_dir: Path, all_event_paths: list[Path], zone_values: np.ndarray | None = None) -> dict:
    metadata = read_metadata(match_dir/'match.xml'); zone_values = learn_zone_values(all_event_paths) if zone_values is None else zone_values
    action_totals, actions = value_match_actions(match_dir/'events.xml', metadata['players'], zone_values)
    spatial_zones, minutes, quality = spatial_match(match_dir/'positions.xml', match_dir/'events.xml', metadata, zone_values)
    output_players=[]
    for i,p in enumerate(metadata['players']):
        if minutes[i] < 1: continue
        action_per90 = action_totals[p.id] * 90/max(minutes[i],1)
        output_players.append({**asdict(p),'minutes':round(float(minutes[i]),1),'action':round(float(action_per90),4),'spatial':round(float(spatial_zones[i].sum()),4),'zones':[round(float(x),5) for x in spatial_zones[i]]})
    return {'metadata':{k:v for k,v in metadata.items() if k!='players'},'provenance':{'source':'DFL / IDSSE','license':'CC BY 4.0','doi':'10.1038/s41597-025-04505-y','computed':True,**quality},'zone_values':zone_values.tolist(),'players':output_players,'actions':actions}
