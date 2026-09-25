'use client';

import { useMemo, useState } from 'react';
import { ArrowDownRight, ArrowRight, CircleGauge, FlaskConical, Layers3, MousePointer2, ShieldCheck } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { actions, match, players, provenance, zoneNames, type MatchAction, type Player } from './data';
import modelReport from './model-report.generated';
import researchV1 from './research-v1.generated';

type PitchLayer = 'spatial' | 'action';
const formatValue = (value: number, digits = 2) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
const totalValue = (player: Player) => player.action + player.spatial;

export function Dashboard() {
  const initialPlayer = players.find((player) => player.name.includes('Kimmich')) ?? players[0];
  const initialCompare = players.find((player) => player.name.includes('Musiala')) ?? players[1];
  const [selectedId, setSelectedId] = useState(initialPlayer.id);
  const [compareId, setCompareId] = useState(initialCompare.id);
  const [team, setTeam] = useState(initialPlayer.team);
  const selected = players.find((player) => player.id === selectedId) ?? players[0];
  const compare = players.find((player) => player.id === compareId) ?? players[1];

  const choosePlayer = (id: string) => {
    const player = players.find((candidate) => candidate.id === id);
    if (!player) return;
    setSelectedId(player.id);
    setTeam(player.team);
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <Header />
      <div className="mx-auto max-w-[1560px] px-4 pb-10 pt-7 sm:px-7 lg:px-10">
        <MatchHeader />
        <Tabs defaultValue="study" className="mt-7">
          <TabsList aria-label="Analysis views" className="lab-tabs">
            <TabsTrigger value="study">Match study</TabsTrigger>
            <TabsTrigger value="compare">Compare players</TabsTrigger>
            <TabsTrigger value="validation">Model validation</TabsTrigger>
            <TabsTrigger value="research">ZCPV v1 research</TabsTrigger>
            <TabsTrigger value="method">Method &amp; limits</TabsTrigger>
          </TabsList>
          <TabsContent value="study" className="mt-5">
            <MatchStudy selected={selected} onSelect={choosePlayer} team={team} setTeam={setTeam} />
          </TabsContent>
          <TabsContent value="compare" className="mt-5">
            <CompareView left={selected} right={compare} setLeft={choosePlayer} setRight={setCompareId} />
          </TabsContent>
          <TabsContent value="validation" className="mt-5"><ValidationView /></TabsContent>
          <TabsContent value="research" className="mt-5"><ResearchV1View /></TabsContent>
          <TabsContent value="method" className="mt-5"><MethodView /></TabsContent>
        </Tabs>
        <footer className="mt-8 flex flex-wrap items-start justify-between gap-3 border-t border-border pt-5 text-xs leading-relaxed text-muted-foreground">
          <p>DFL / IDSSE data · CC BY 4.0 · ZCPV values are experimental and are not official DFL ratings.</p>
          <a className="link" href={`https://doi.org/${provenance.doi}`}>Dataset methodology <ArrowRight size={12} /></a>
        </footer>
      </div>
    </main>
  );
}

function Header() {
  return <header className="border-b border-border bg-card"><div className="mx-auto flex h-16 max-w-[1560px] items-center justify-between px-4 sm:px-7 lg:px-10">
    <div className="flex items-center gap-3"><span className="brand-mark">Z</span><div><p className="text-sm font-semibold tracking-[-.01em]">ZCPV Research Lab</p><p className="text-[11px] text-muted-foreground">Counterfactual football analysis</p></div></div>
    <div className="status"><span /> Verified public match data</div>
  </div></header>;
}

function MatchHeader() {
  const [homeScore, awayScore] = match.score.split(':');
  return <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
    <div><p className="eyebrow">Bundesliga · Matchday {match.matchday} · 27 May 2023</p><div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2"><h1 className="display-title">{match.teams[0]}</h1><div className="scoreline"><b>{homeScore}</b><span>–</span><b>{awayScore}</b></div><h1 className="display-title">{match.teams[1]}</h1></div><p className="mt-3 max-w-3xl text-sm leading-6 text-muted-foreground">A single-match study of how actions and positioning changed possession threat. Select a player, then inspect the evidence behind the score.</p></div>
    <div className="flex flex-wrap gap-2"><Fact label="Tracking" value={`${provenance.analysis_hz} Hz`} icon={<CircleGauge size={15} />} /><Fact label="Live frames" value={provenance.live_frames.toLocaleString()} icon={<Layers3 size={15} />} /><Fact label="Coverage" value={`${Math.round(provenance.coverage * 100)}%`} icon={<ShieldCheck size={15} />} /></div>
  </section>;
}

function MatchStudy({ selected, onSelect, team, setTeam }: { selected: Player; onSelect: (id: string) => void; team: string; setTeam: (team: string) => void }) {
  const ranked = useMemo(() => players.filter((player) => player.team === team && player.minutes >= 30).sort((a, b) => totalValue(b) - totalValue(a)), [team]);
  const playerActions = useMemo(() => actions.filter((action) => action.player === selected.id).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)), [selected.id]);
  const [actionIndex, setActionIndex] = useState(0);
  const [layer, setLayer] = useState<PitchLayer>('spatial');
  const selectedAction = playerActions[actionIndex] ?? playerActions[0];
  const pickPlayer = (id: string) => { setActionIndex(0); setLayer('spatial'); onSelect(id); };

  return <div className="workspace-grid">
    <aside className="panel overflow-hidden"><div className="panel-heading"><div><p className="eyebrow">Legacy v0 match contribution</p><h2 className="mt-1 font-semibold">Player index</h2></div><TeamToggle team={team} setTeam={setTeam} /></div><div className="border-b border-border px-4 py-3 text-xs text-muted-foreground">Players with at least 30 minutes · sorted by descriptive v0 score / 90</div><div>{ranked.map((player, index) => <PlayerRow key={player.id} player={player} index={index} active={selected.id === player.id} onClick={() => pickPlayer(player.id)} />)}</div></aside>
    <section className="space-y-4">
      <div className="panel overflow-hidden"><div className="flex flex-wrap items-start justify-between gap-4 border-b border-border p-5"><div><p className="eyebrow">Evidence view</p><h2 className="mt-1 text-xl font-semibold">{layer === 'spatial' ? 'Where positioning carried value' : 'Selected on-ball action'}</h2></div><div className="segmented" aria-label="Pitch layer"><button className={layer === 'spatial' ? 'active' : ''} onClick={() => setLayer('spatial')}>Spatial value</button><button className={layer === 'action' ? 'active' : ''} onClick={() => setLayer('action')} disabled={!selectedAction}>Action</button></div></div><div className="p-4 sm:p-6"><Pitch values={selected.zones} action={layer === 'action' ? selectedAction : undefined} /><div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground"><p>{layer === 'spatial' ? `Estimated team-control value lost when ${shortName(selected.name)} is removed.` : 'Origin and destination are reconstructed from the official synchronized event feed.'}</p><span>Attacking direction <ArrowRight className="inline" size={13} /></span></div></div></div>
      <ActionLedger actions={playerActions} selectedIndex={actionIndex} onSelect={(index) => { setActionIndex(index); setLayer('action'); }} />
    </section>
    <aside className="space-y-4"><PlayerBrief player={selected} actions={playerActions} selectedAction={selectedAction} /><div className="note-card"><div className="flex gap-3"><FlaskConical size={17} /><div><b>Interpret carefully</b><p>This is one match, not a scouting grade. Per-90 rates amplify short appearances and confidence intervals are not yet available.</p></div></div></div></aside>
  </div>;
}

function PlayerRow({ player, index, active, onClick }: { player: Player; index: number; active: boolean; onClick: () => void }) {
  const score = totalValue(player);
  return <button onClick={onClick} className={`player-row ${active ? 'active' : ''}`} aria-label={`Inspect ${player.name}`}><span className="rank">{String(index + 1).padStart(2, '0')}</span><span className="shirt">{player.number}</span><span className="min-w-0 flex-1"><span className="flex items-baseline justify-between gap-2"><b className="truncate text-sm">{player.name}</b><b className={score < 0 ? 'negative' : ''}>{formatValue(score)}</b></span><span className="mt-1 flex justify-between text-[11px] text-muted-foreground"><span>{player.role} · {player.minutes.toFixed(0)} min</span><span>{formatValue(player.action)} ball · {formatValue(player.spatial)} space</span></span></span><span className="text-muted-foreground">›</span></button>;
}

function PlayerBrief({ player, actions: playerActions, selectedAction }: { player: Player; actions: MatchAction[]; selectedAction?: MatchAction }) {
  const score = totalValue(player);
  const actionShare = Math.round((Math.abs(player.action) / Math.max(0.001, Math.abs(player.action) + Math.abs(player.spatial))) * 100);
  const sample = player.minutes >= 75 ? 'Full-match sample' : player.minutes >= 45 ? 'Moderate sample' : 'Limited sample';
  return <section className="panel p-5"><div className="flex items-start justify-between gap-3"><div><p className="eyebrow">Selected player</p><h2 className="mt-2 text-2xl font-semibold">{player.name}</h2><p className="mt-1 text-sm text-muted-foreground">#{player.number} · {player.role} · {player.team}</p></div><span className="sample-tag">{sample}</span></div><div className="score-block"><p>Legacy v0 / 90</p><strong className={score < 0 ? 'negative' : ''}>{formatValue(score)}</strong><span>Descriptive single-match contribution</span></div><div className="metric-pair"><Metric label="On-ball" value={player.action} detail={`${actionShare}% of absolute score`} /><Metric label="Off-ball space" value={player.spatial} detail={`${100 - actionShare}% of absolute score`} /></div><div className="mt-5 border-t border-border pt-5"><div className="flex items-center justify-between"><p className="eyebrow">Action evidence</p><span className="text-xs text-muted-foreground">{playerActions.length} valued actions</span></div>{selectedAction ? <div className="mt-3"><div className="flex items-center justify-between"><b>{selectedAction.minute}′ · {selectedAction.type}</b><b className={selectedAction.value < 0 ? 'negative' : 'positive'}>{formatValue(selectedAction.value, 3)}</b></div><p className="mt-2 text-sm leading-6 text-muted-foreground">{explainAction(selectedAction)}</p></div> : <p className="mt-3 text-sm text-muted-foreground">No valued on-ball event is attributed to this player in the feed.</p>}</div></section>;
}

function ActionLedger({ actions: playerActions, selectedIndex, onSelect }: { actions: MatchAction[]; selectedIndex: number; onSelect: (index: number) => void }) {
  return <section className="panel overflow-hidden"><div className="panel-heading"><div><p className="eyebrow">Traceable evidence</p><h2 className="mt-1 font-semibold">Highest-impact actions</h2></div><MousePointer2 size={16} className="text-muted-foreground" /></div>{playerActions.length ? <div className="action-table"><div className="action-head"><span>Minute</span><span>Action</span><span>Movement</span><span>Value</span></div>{playerActions.slice(0, 7).map((action, index) => <button key={`${action.minute}-${index}`} className={selectedIndex === index ? 'active' : ''} onClick={() => onSelect(index)}><span>{action.minute}′</span><b>{action.type}</b><span>{zoneNames[action.from]} <ArrowDownRight size={12} /> {zoneNames[action.to]}</span><strong className={action.value < 0 ? 'negative' : 'positive'}>{formatValue(action.value, 3)}</strong></button>)}</div> : <p className="p-5 text-sm text-muted-foreground">No attributed actions are available for this player.</p>}</section>;
}

function Pitch({ values, action }: { values: number[]; action?: MatchAction }) {
  const max = Math.max(...values.map(Math.abs), 0.001); const start = action ? zoneCenter(action.from) : null; const end = action ? zoneCenter(action.to) : null;
  return <svg viewBox="0 0 630 408" className="pitch" aria-label={action ? 'Selected action origin and destination' : 'Spatial contribution heatmap'}><defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#f5f0e6" /></marker><pattern id="grass" width="140" height="408" patternUnits="userSpaceOnUse"><rect width="70" height="408" fill="#315f47"/><rect x="70" width="70" height="408" fill="#2e5943"/></pattern></defs><rect width="630" height="408" rx="8" fill="url(#grass)" />{!action && values.map((value, index) => { const xBand = Math.floor(index / 3); const yBand = index % 3; return <rect key={index} x={20 + xBand * 147.5} y={20 + yBand * 122.66} width="147.5" height="122.66" fill={value >= 0 ? '#f1c75b' : '#d05a4e'} opacity={0.06 + 0.66 * Math.abs(value) / max} />; })}<g fill="none" stroke="rgba(255,255,255,.78)" strokeWidth="2"><rect x="20" y="20" width="590" height="368"/><path d="M315 20v368"/><circle cx="315" cy="204" r="49"/><circle cx="315" cy="204" r="2" fill="white"/><path d="M20 115h86v178H20m590-178h-86v178h86M20 151h34v106H20m590-106h-34v106h34"/><path d="M106 167a49 49 0 0 1 0 74m418-74a49 49 0 0 0 0 74" opacity=".7"/></g>{action && start && end && <g><line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke="#f5f0e6" strokeWidth="5" markerEnd="url(#arrowhead)" /><circle cx={start.x} cy={start.y} r="12" fill="#17211b" stroke="#f5f0e6" strokeWidth="3" /><circle cx={end.x} cy={end.y} r="9" fill={action.value >= 0 ? '#f1c75b' : '#d05a4e'} stroke="#17211b" strokeWidth="3" /><text x={start.x} y={start.y - 20} textAnchor="middle" className="pitch-label">{action.minute}′ {action.type}</text></g>}</svg>;
}

function CompareView({ left, right, setLeft, setRight }: { left: Player; right: Player; setLeft: (id: string) => void; setRight: (id: string) => void }) {
  return <section className="panel p-5 sm:p-7"><div className="max-w-2xl"><p className="eyebrow">Within-match comparison</p><h2 className="mt-2 text-2xl font-semibold">Separate visible actions from spatial influence</h2><p className="mt-2 text-sm leading-6 text-muted-foreground">Use this as a descriptive comparison of one match. It is not a role-adjusted or season-level player rating.</p></div><div className="mt-7 grid gap-5 lg:grid-cols-2"><CompareCard player={left} onChange={setLeft} /><CompareCard player={right} onChange={setRight} /></div></section>;
}

function CompareCard({ player, onChange }: { player: Player; onChange: (id: string) => void }) {
  const score = totalValue(player);
  return <article className="compare-card"><Select value={player.id} onValueChange={(value) => value && onChange(value)}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{players.map((candidate) => <SelectItem key={candidate.id} value={candidate.id}>{candidate.name} · {candidate.role}</SelectItem>)}</SelectContent></Select><div className="mt-7 flex items-end justify-between"><div><p className="eyebrow">Legacy v0 / 90</p><p className={`mt-1 text-5xl font-semibold tracking-[-.06em] ${score < 0 ? 'negative' : ''}`}>{formatValue(score)}</p></div><span className="sample-tag">{player.minutes.toFixed(0)} minutes</span></div><div className="mt-7 space-y-5"><CompareBar label="On-ball action value" value={player.action} max={1.5} /><CompareBar label="Off-ball spatial value" value={player.spatial} max={3} /><CompareBar label="Combined" value={score} max={4} /></div></article>;
}

function ValidationView() {
  const { dataset, vaep, xt, notes } = modelReport;
  return <div className="space-y-5">
    <section className="panel p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-5">
        <div className="max-w-2xl"><p className="eyebrow">Out-of-sample benchmark</p><h2 className="mt-2 text-2xl font-semibold">World Cup 2022 event baselines</h2><p className="mt-2 text-sm leading-6 text-muted-foreground">A trained 16 × 12 xT surface now supplies the zone values in the DFL score. The VAEP-style probability model remains an independent event-data benchmark.</p></div>
        <a className="source-link" href={dataset.repository}>StatsBomb Open Data <ArrowRight size={13} /></a>
      </div>
      <div className="split-strip">
        <div style={{width:`${dataset.train_matches / dataset.matches * 100}%`}}><b>{dataset.train_matches} matches</b><span>Training · {dataset.train_actions.toLocaleString()} actions</span></div>
        <div className="test"><b>{dataset.test_matches} matches</b><span>Held out · {dataset.test_actions.toLocaleString()} actions</span></div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">Chronological split · test period begins {dataset.test_start} · no held-out match is used to fit xT or VAEP.</p>
    </section>

    <div className="validation-grid">
      <ValidationCard title="Scoring probability" subtitle="Goal by the team in possession within 10 actions" result={vaep.score} />
      <ValidationCard title="Conceding probability" subtitle="Opponent goal within 10 actions" result={vaep.concede} />
      <section className="panel p-5"><p className="eyebrow">xT baseline</p><h3 className="mt-2 text-lg font-semibold">{xt.grid[0]} × {xt.grid[1]} observed-event grid</h3><div className="mt-5 grid grid-cols-2 gap-2"><MiniResult label="Populated zones" value={`${xt.nonzero_zones}/${xt.grid[0] * xt.grid[1]}`} /><MiniResult label="Held-out actions" value={xt.test_actions_valued.toLocaleString()} /><MiniResult label="Peak zone value" value={xt.max_zone_value.toFixed(3)} /><MiniResult label="Mean |action value|" value={xt.mean_absolute_action_value.toFixed(4)} /></div><p className="mt-4 text-xs leading-5 text-muted-foreground">The grid is fitted on observed passes, carries, dribbles, shots, goals and turnovers—without the former hand-shaped goal prior.</p></section>
    </div>

    <section className="panel p-5"><div className="grid gap-5 md:grid-cols-[180px_1fr]"><div><p className="eyebrow">Reading the result</p><p className="mt-2 text-sm font-semibold">Useful, not conclusive</p></div><ul className="validation-notes">{notes.map((note) => <li key={note}>{note}</li>)}</ul></div></section>
  </div>;
}

function ResearchV1View() {
  const stages = [
    ['Canonical DFL ingestion', 'Implemented', 'Events, lineups, sampled tracking, checksums and per-match quality reports'],
    ['15-second npxG targets', 'Blocked', 'The DFL sample does not provide xG; the pipeline refuses to substitute the legacy formula'],
    ['Spatial player evidence', 'Implemented', 'Receiving feasibility, pressure episodes, lane coverage and transition-protection interfaces'],
    ['Offensive / defensive impact', 'Ready', 'Duration-weighted regularized estimator with lagged profiles and lineup-only baseline support'],
    ['Published player ratings', 'Unavailable', 'Seven matches cannot identify credible out-of-time player effects'],
  ];
  return <div className="space-y-5">
    <section className="panel p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-5">
        <div className="max-w-3xl"><p className="eyebrow">Research pipeline · {researchV1.model_version}</p><h2 className="mt-2 text-2xl font-semibold">Built to fail honestly when evidence is insufficient</h2><p className="mt-2 text-sm leading-6 text-muted-foreground">V1 is specified as offensive and defensive non-penalty xG impact per 90. This dataset can exercise ingestion and feature engineering, but it cannot yet support a defensible rating table.</p></div>
        <span className="sample-tag">{researchV1.status.replace('_', ' ')}</span>
      </div>
      <div className="mt-6 rounded-xl border border-border bg-muted/30 p-4"><p className="text-sm font-semibold">Why the ratings are blank</p><p className="mt-1 text-sm leading-6 text-muted-foreground">{researchV1.reason}</p></div>
    </section>
    <section className="panel overflow-hidden"><div className="panel-heading"><div><p className="eyebrow">Implementation status</p><h2 className="mt-1 font-semibold">V1 evidence chain</h2></div></div><div className="divide-y divide-border">{stages.map(([title, status, detail]) => <div key={title} className="grid gap-2 px-5 py-4 md:grid-cols-[220px_120px_1fr]"><b className="text-sm">{title}</b><span className="text-xs uppercase tracking-[.12em] text-muted-foreground">{status}</span><p className="text-sm text-muted-foreground">{detail}</p></div>)}</div></section>
    <section className="grid gap-5 md:grid-cols-3"><div className="panel p-5"><p className="eyebrow">Target unit</p><p className="mt-2 text-lg font-semibold">npxG impact / 90</p><p className="mt-2 text-xs leading-5 text-muted-foreground">Not goals, wins or replacement value.</p></div><div className="panel p-5"><p className="eyebrow">Public reference</p><p className="mt-2 text-lg font-semibold">Training-average player</p><p className="mt-2 text-xs leading-5 text-muted-foreground">Exposure weighted; not automatically position relative.</p></div><div className="panel p-5"><p className="eyebrow">Available inputs</p><p className="mt-2 text-lg font-semibold">{researchV1.data_summary.dfl_matches} DFL matches</p><p className="mt-2 text-xs leading-5 text-muted-foreground">Good for software audits and exploratory diagnostics, not season rankings.</p></div></section>
  </div>;
}

function ValidationCard({ title, subtitle, result }: { title: string; subtitle: string; result: typeof modelReport.vaep.score }) {
  return <section className="panel p-5"><p className="eyebrow">VAEP-style baseline</p><h3 className="mt-2 text-lg font-semibold">{title}</h3><p className="mt-1 min-h-10 text-xs leading-5 text-muted-foreground">{subtitle}</p><div className="validation-score"><div><span>Brier score</span><b>{result.brier.toFixed(5)}</b></div><div><span>vs constant baseline</span><b className="positive">{result.brier_improvement_pct.toFixed(1)}% better</b></div></div><div className="mt-4 grid grid-cols-3 gap-2"><MiniResult label="ROC AUC" value={result.roc_auc.toFixed(3)} /><MiniResult label="Log loss" value={result.log_loss.toFixed(4)} /><MiniResult label="Calibration error" value={result.calibration_error.toFixed(4)} /></div><p className="mt-4 text-[11px] text-muted-foreground">Evaluated on {result.events.toLocaleString()} unseen action states · positive rate {(result.positive_rate * 100).toFixed(2)}%</p></section>;
}

function MiniResult({ label, value }: { label: string; value: string }) { return <div className="mini-result"><span>{label}</span><b>{value}</b></div>; }

function MethodView() {
  const stages = [
    { number: '01', title: 'Observed actions', copy: 'The legacy v0 demo values successful moves by zone change. Its shot and defensive values remain heuristic and are not v1 training targets.' },
    { number: '02', title: 'Pitch control', copy: 'Player position, velocity and reaction time estimate which team can reach every cell of a 32 × 21 grid first.' },
    { number: '03', title: 'Counterfactual removal', copy: 'Each player is removed in turn. The drop in their team’s zone-weighted control is assigned as spatial contribution.' },
    { number: '04', title: 'Possession aggregation', copy: 'Frames are averaged within possessions so long possessions do not receive more weight simply because they last longer.' },
  ];
  return <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(320px,.65fr)]"><section className="panel p-5 sm:p-7"><p className="eyebrow">Legacy v0 match demo</p><h2 className="mt-2 text-2xl font-semibold">What the visible match score measures</h2><p className="mt-2 text-sm leading-6 text-muted-foreground">This method remains available for continuity. It is not the new v1 impact estimator.</p><div className="mt-7 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2">{stages.map((stage) => <div key={stage.number} className="method-step"><span>{stage.number}</span><h3>{stage.title}</h3><p>{stage.copy}</p></div>)}</div></section><aside className="space-y-5"><section className="panel p-5"><p className="eyebrow">Known limitations</p><ul className="plain-list"><li>World Cup event values may not transfer perfectly to Bundesliga play.</li><li>No bootstrap uncertainty intervals yet.</li><li>Shots and defensive actions do not yet use the trained VAEP model.</li><li>Legacy v0 is descriptive, not a causal player rating.</li><li>Per-90 values are unstable for substitutes.</li></ul></section><section className="panel p-5"><p className="eyebrow">Compute benchmark</p><div className="benchmark"><b>47.5 ms</b><span>CUDA · RTX 3080</span></div><div className="mt-4 grid grid-cols-2 gap-2 text-sm"><div className="stat-cell"><span>CPU optimized</span><b>6.45 s</b></div><div className="stat-cell"><span>Speedup</span><b>135.7×</b></div></div><p className="mt-4 text-xs leading-5 text-muted-foreground">Measured on the full Köln–Bayern match at 5 Hz. This is a compute benchmark, not evidence of model accuracy.</p></section></aside></div>;
}

function TeamToggle({ team, setTeam }: { team: string; setTeam: (team: string) => void }) { const other = team === match.teams[0] ? match.teams[1] : match.teams[0]; return <button className="quiet-button" onClick={() => setTeam(other)}>Switch team</button>; }
function Metric({ label, value, detail }: { label: string; value: number; detail: string }) { return <div><span>{label}</span><b className={value < 0 ? 'negative' : ''}>{formatValue(value)}</b><small>{detail}</small></div>; }
function CompareBar({ label, value, max }: { label: string; value: number; max: number }) { return <div><div className="mb-2 flex justify-between text-sm"><span>{label}</span><b className={value < 0 ? 'negative' : ''}>{formatValue(value)}</b></div><div className="bar-track"><span className={value < 0 ? 'negative-bar' : ''} style={{ width: `${Math.min(100, Math.abs(value) / max * 100)}%` }} /></div></div>; }
function Fact({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) { return <div className="fact"><i>{icon}</i><div><span>{label}</span><b>{value}</b></div></div>; }
function zoneCenter(index: number) { const xBand = Math.floor(index / 3); const yBand = index % 3; return { x: 20 + (xBand + 0.5) * 147.5, y: 20 + (yBand + 0.5) * 122.66 }; }
function shortName(name: string) { return name.split(' ').at(-1) ?? name; }
function explainAction(action: MatchAction) { const direction = action.to === action.from ? `within ${zoneNames[action.from]}` : `from ${zoneNames[action.from]} to ${zoneNames[action.to]}`; if (action.value > 0) return `This ${action.type.toLowerCase()} increased the modelled possession value ${direction}. The number is the event-level change, before per-90 normalization.`; if (action.value < 0) return `This ${action.type.toLowerCase()} reduced the modelled possession value ${direction}. Negative events remain visible rather than being hidden by the player total.`; return `This ${action.type.toLowerCase()} produced no measured change in the current 12-zone action model.`; }
