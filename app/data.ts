import realMatch from '../public/data/dfl-match.json';

export type Player = {
  id: string; name: string; number: number; role: string; team: string;
  minutes: number; action: number; spatial: number; zones: number[];
};

const roles: Record<string,string> = { TW:'GK', IVR:'CB', IVL:'CB', IVZ:'CB', RV:'RB', LV:'LB', DMR:'DM', DML:'DM', ZO:'AM', RA:'RW', LA:'LW', ORM:'RW', OLM:'LW', STZ:'ST', HR:'RM', HL:'LM', SUB:'SUB' };

export const match = realMatch.metadata;
export const provenance = realMatch.provenance;
export const players: Player[] = realMatch.players.map(p => ({
  id:p.id, name:p.name, number:p.number, role:roles[p.role] ?? p.role, team:p.team_name,
  minutes:p.minutes, action:p.action, spatial:p.spatial, zones:p.zones,
}));

export const actions = realMatch.actions.map(a => ({...a, detail:a.detail ?? `${a.type} from zone ${a.from + 1} to ${a.to + 1}`}));
export const zoneNames = ['Own L','Own C','Own R','Def-mid L','Def-mid C','Def-mid R','Att-mid L','Att-mid C','Att-mid R','Final L','Final C','Final R'];
