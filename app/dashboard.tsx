'use client';

import { useEffect, useMemo, useState } from 'react';
import { Activity, BarChart3, Check, Cpu, Info, Radio, ScanLine, Timer, Zap } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { actions, benchmarks, match, players, provenance, zoneNames, type Player } from './data';

declare global {
  interface Document { modelContext?: { registerTool: (tool: Record<string, unknown>, options?: { signal?: AbortSignal }) => void | Promise<void> } }
}

export function Dashboard() {
  const initialPlayer = players.find(p=>p.name.includes('Sané')) ?? players[0];
  const initialCompare = players.find(p=>p.name.includes('Musiala')) ?? players[1];
  const [selectedId, setSelectedId] = useState(initialPlayer.id);
  const [compareId, setCompareId] = useState(initialCompare.id);
  const [team, setTeam] = useState<string>(initialPlayer.team);
  const selected = players.find(p => p.id === selectedId) ?? players[0];
  const compare = players.find(p => p.id === compareId) ?? players[11];

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const selectPlayer = (input: unknown) => {
      const playerId = typeof input === 'object' && input ? (input as { playerId?: unknown }).playerId : undefined;
      const player = players.find(p => p.id === playerId);
      if (!player) throw new Error('Unknown playerId');
      setSelectedId(player.id); setTeam(player.team);
      return { playerId: player.id, playerName: player.name, zcpv: Number((player.action + player.spatial).toFixed(2)) };
    };
    try {
      void Promise.resolve(context.registerTool({
        name:'select_player', title:'Select player', description:'Select a player in the visible ZCPV dashboard.',
        inputSchema:{ type:'object', properties:{playerId:{type:'string', enum:players.map(p => p.id)}}, required:['playerId'], additionalProperties:false },
        annotations:{readOnlyHint:false, untrustedContentHint:false}, execute:selectPlayer,
      }, { signal:lifecycle.signal })).catch(() => undefined);
    } catch { /* WebMCP is optional in unsupported browsers. */ }
    return () => lifecycle.abort();
  }, []);

  return <main className="min-h-screen bg-background px-4 py-4 text-foreground sm:px-6 lg:px-8">
    <Header />
    <section className="mx-auto mt-6 max-w-[1500px]">
      <MatchHeader />
      <Tabs defaultValue="match" className="mt-6">
        <TabsList aria-label="Dashboard views" className="h-auto rounded-xl border border-white/10 bg-white/[.035] p-1">
          <TabsTrigger value="match">Match lab</TabsTrigger><TabsTrigger value="compare">Compare</TabsTrigger><TabsTrigger value="method">Model & benchmarks</TabsTrigger>
        </TabsList>
        <TabsContent value="match" className="mt-5"><MatchView selected={selected} setSelectedId={setSelectedId} team={team} setTeam={setTeam} /></TabsContent>
        <TabsContent value="compare" className="mt-5"><CompareView left={selected} right={compare} setLeft={setSelectedId} setRight={setCompareId} /></TabsContent>
        <TabsContent value="method" className="mt-5"><MethodView /></TabsContent>
      </Tabs>
      <footer className="mt-6 border-t border-white/10 py-5 text-xs leading-relaxed text-muted-foreground">Data: Deutsche Fußball Liga (DFL), licensed CC BY 4.0. Methodology dataset: Bassek, Rein, Weber &amp; Memmert (2025), <a className="text-lime-300 hover:underline" href={`https://doi.org/${provenance.doi}`}>Scientific Data</a>. ZCPV values are research-prototype outputs, not official DFL ratings.</footer>
    </section>
  </main>;
}

function Header() {
  return <header className="mx-auto flex max-w-[1500px] items-center justify-between border-b border-white/10 pb-4">
    <div className="flex items-center gap-3"><span className="grid size-10 place-items-center rounded-xl bg-lime-300 text-slate-950"><Activity size={20} /></span><div><p className="font-semibold tracking-tight">ZCPV Lab</p><p className="text-xs text-muted-foreground">Spatial player value</p></div></div>
    <div className="flex items-center gap-2 rounded-full border border-lime-300/20 bg-lime-300/5 px-3 py-2 text-xs text-lime-200"><Radio size={13} /> Official DFL tracking</div>
  </header>;
}

function MatchHeader() {
  const [homeScore,awayScore]=match.score.split(':');
  return <div className="flex flex-wrap items-end justify-between gap-4">
    <div><p className="eyebrow">Bundesliga · Matchday {match.matchday} · May 27, 2023 · Full time</p><h1 className="mt-2 text-3xl font-semibold tracking-[-.04em] sm:text-4xl">{match.teams[0]} <span className="text-muted-foreground">{homeScore}</span> — <span className="text-muted-foreground">{awayScore}</span> {match.teams[1]}</h1></div>
    <div className="flex gap-2"><MiniMetric icon={<Cpu size={15}/>} label="Engine" value="Measured CPU"/><MiniMetric icon={<ScanLine size={15}/>} label="Grid" value="32 × 21"/><MiniMetric icon={<Timer size={15}/>} label="Tracking" value={`${provenance.analysis_hz} Hz of ${provenance.source_hz} Hz`}/></div>
  </div>;
}

function MatchView({selected,setSelectedId,team,setTeam}:{selected:Player;setSelectedId:(id:string)=>void;team:string;setTeam:(t:string)=>void}) {
  const ranked = useMemo(() => players.filter(p=>p.team===team&&p.minutes>=30).sort((a,b)=>(b.action+b.spatial)-(a.action+a.spatial)),[team]);
  return <div className="grid gap-5 xl:grid-cols-[350px_minmax(0,1fr)]">
    <aside className="panel overflow-hidden">
      <div className="flex items-end justify-between border-b border-white/10 px-5 py-4"><div><p className="eyebrow">Match ranking · ≥30 min</p><h2 className="mt-1 text-lg font-semibold">{team}</h2></div><button onClick={()=>setTeam(team===match.teams[0]?match.teams[1]:match.teams[0])} className="rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-white/5 hover:text-white">Switch team</button></div>
      <div>{ranked.map((p,i)=><PlayerRow key={p.id} player={p} index={i} active={selected.id===p.id} onClick={()=>setSelectedId(p.id)}/>)}</div>
      <div className="border-t border-white/10 p-4 text-xs leading-relaxed text-muted-foreground"><Info size={13} className="mr-1 inline"/> Scores are per 90. Cyan is on-ball action value; lime is off-ball spatial value.</div>
    </aside>
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,.8fr)_minmax(380px,1.2fr)]"><ScoreBreakdown player={selected}/><PitchPanel player={selected}/></div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,.85fr)]"><Timeline player={selected}/><ZoneGrid player={selected}/></div>
    </div>
  </div>;
}

function PlayerRow({player,index,active,onClick}:{player:Player;index:number;active:boolean;onClick:()=>void}) {
  const score=player.action+player.spatial; const pct=Math.max(5,Math.abs(player.action)/(Math.abs(player.action)+Math.abs(player.spatial))*100);
  return <button aria-label={`View ${player.name}, ZCPV ${score.toFixed(2)}`} onClick={onClick} className={`block w-full border-b border-white/[.06] px-5 py-3.5 text-left transition last:border-0 ${active?'bg-lime-300/[.08]':'hover:bg-white/[.035]'}`}>
    <span className="flex items-center gap-3"><span className="w-4 text-xs text-muted-foreground">{index+1}</span><span className="grid size-8 place-items-center rounded-full bg-slate-700 text-[11px] font-bold">{player.number}</span><span className="min-w-0 flex-1"><span className="flex justify-between gap-2"><span className="truncate text-sm font-medium">{player.name} <span className="ml-1 text-xs font-normal text-muted-foreground">{player.role}</span></span><span className="font-mono text-sm text-lime-300">+{score.toFixed(2)}</span></span><span className="mt-2 flex h-1.5 overflow-hidden rounded-full bg-white/8"><span className="bg-cyan-400" style={{width:`${pct}%`}}/><span className="flex-1 bg-lime-300"/></span></span></span>
  </button>;
}

function ScoreBreakdown({player}:{player:Player}) { const total=player.action+player.spatial; const actionPct=Math.round(Math.abs(player.action)/(Math.abs(player.action)+Math.abs(player.spatial))*100); return <section className="panel p-5 sm:p-6">
  <div className="flex items-start justify-between gap-3"><div><p className="eyebrow">Selected · {player.role} · {player.minutes} min</p><h2 className="mt-2 text-2xl font-semibold">{player.name}</h2><p className="mt-1 text-sm text-muted-foreground">#{player.number} · {player.team}</p></div><div className="text-right"><p className="text-4xl font-semibold tracking-[-.05em] text-lime-300">+{total.toFixed(2)}</p><p className="text-xs text-muted-foreground">ZCPV / 90</p></div></div>
  <div className="mt-6 grid grid-cols-2 gap-3"><ScoreCard label="Action value" value={player.action}/><ScoreCard label="Spatial value" value={player.spatial} spatial/></div>
  <div className="mt-6"><div className="mb-2 flex justify-between text-sm"><span>Magnitude split</span><span className="text-muted-foreground">{actionPct}% / {100-actionPct}%</span></div><div className="flex h-2 overflow-hidden rounded-full bg-white/8"><span className="bg-cyan-400" style={{width:`${actionPct}%`}}/><span className="flex-1 bg-lime-300"/></div></div>
</section>; }

function ScoreCard({label,value,spatial=false}:{label:string;value:number;spatial?:boolean}) { return <div className="rounded-xl border border-white/10 bg-black/15 p-4"><div className={`mb-3 h-1 w-8 rounded-full ${spatial?'bg-lime-300':'bg-cyan-400'}`}/><p className="text-sm text-muted-foreground">{label}</p><p className="mt-1 text-2xl font-semibold">{value>=0?'+':''}{value.toFixed(2)}</p></div>; }

function PitchPanel({player}:{player:Player}) { return <section className="panel p-5 sm:p-6"><div className="mb-4 flex items-start justify-between"><div><p className="eyebrow">Leave-one-out control</p><h2 className="mt-1 text-lg font-semibold">Zone value lost without {player.name.split(' ')[0]}</h2></div><span className="rounded-full bg-lime-300/10 px-2 py-1 text-xs text-lime-200">Attacking direction →</span></div><Pitch values={player.zones}/></section>; }

function Pitch({values}:{values:number[]}) { const max=Math.max(...values,.001); return <svg viewBox="0 0 520 340" aria-label="Pitch heatmap of leave-one-out control contribution" className="w-full rounded-lg border border-white/10 bg-[#102f2b]">
  <title>Pitch heatmap of leave-one-out control contribution</title>
  <g>{values.map((value,index)=>{const xband=Math.floor(index/3),yband=index%3; return <rect key={index} x={15+xband*122.5} y={15+yband*103.33} width="122.5" height="103.33" fill="#bef264" opacity={.04+.68*value/max}/>})}</g>
  <g fill="none" stroke="rgba(255,255,255,.38)" strokeWidth="2"><rect x="15" y="15" width="490" height="310"/><path d="M260 15v310"/><circle cx="260" cy="170" r="42"/><circle cx="260" cy="170" r="2" fill="white"/><path d="M15 95h70v150H15m490-150h-70v150h70M15 125h28v90H15m490-90h-28v90h28"/></g>
</svg>; }

function Timeline({player}:{player:Player}) { const own=actions.filter(a=>a.player===player.id).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,8).sort((a,b)=>a.minute-b.minute); return <section className="panel p-5 sm:p-6"><div><p className="eyebrow">Action timeline</p><h2 className="mt-1 text-lg font-semibold">Highest-impact actions</h2></div>{own.length?<div className="relative mt-6"><div className="absolute left-0 right-0 top-[7px] h-px bg-white/15"/><div className="relative flex justify-between">{own.map((a,i)=><div key={`${a.minute}-${i}`} className="group relative flex w-5 flex-col items-center"><span className={`size-3 rounded-full ring-4 ring-[#0c1b18] ${a.value>=0?'bg-lime-300':'bg-rose-400'}`}/><span className="mt-3 text-[11px] text-muted-foreground">{a.minute}′</span><span className="pointer-events-none absolute bottom-8 z-20 hidden w-48 -translate-x-[42%] rounded-lg border border-white/10 bg-slate-950 p-3 text-left shadow-xl group-hover:block"><b className="block text-xs">{a.type} · {a.value>0?'+':''}{a.value.toFixed(2)}</b><span className="mt-1 block text-xs leading-relaxed text-slate-400">{a.detail} · zone {a.from+1} → {a.to+1}</span></span></div>)}</div></div>:<p className="mt-6 text-sm text-muted-foreground">No valued on-ball actions in this sample.</p>}<div className="mt-7 rounded-lg bg-white/[.035] px-4 py-3 text-xs text-muted-foreground">Derived from synchronized official DFL events. Hover a marker for attribution.</div></section>; }

function ZoneGrid({player}:{player:Player}) { const max=Math.max(...player.zones); return <section className="panel p-5 sm:p-6"><p className="eyebrow">Spatial decomposition</p><h2 className="mt-1 text-lg font-semibold">Zone contribution</h2><div className="mt-5 grid grid-cols-4 gap-2">{player.zones.map((v,i)=><div key={zoneNames[i]} title={`${zoneNames[i]}: +${v.toFixed(3)}`} className="aspect-[1.3] rounded-md border border-white/10 p-2" style={{background:`rgba(190,242,100,${.04+.55*v/max})`}}><span className="text-[10px] text-white/65">{zoneNames[i]}</span><b className="block text-xs">+{v.toFixed(2)}</b></div>)}</div></section>; }

function CompareView({left,right,setLeft,setRight}:{left:Player;right:Player;setLeft:(id:string)=>void;setRight:(id:string)=>void}) { return <div className="panel p-5 sm:p-7"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Scouting lens</p><h2 className="mt-1 text-2xl font-semibold">Player comparison</h2></div><p className="max-w-md text-sm leading-relaxed text-muted-foreground">Separate visible actions from spatial influence. Values are normalized per 90 for comparable match samples.</p></div><div className="mt-7 grid gap-5 lg:grid-cols-[1fr_auto_1fr]"><CompareCard player={left} onChange={setLeft}/><div className="hidden place-items-center text-muted-foreground lg:grid">vs</div><CompareCard player={right} onChange={setRight}/></div></div>; }

function CompareCard({player,onChange}:{player:Player;onChange:(id:string)=>void}) { const total=player.action+player.spatial; return <div className="rounded-2xl border border-white/10 bg-black/15 p-5"><Select value={player.id} onValueChange={(value)=>{if(value) onChange(value)}}><SelectTrigger aria-label={`Choose player, currently ${player.name}`} className="w-full bg-white/[.035]"><SelectValue/></SelectTrigger><SelectContent>{players.map(p=><SelectItem key={p.id} value={p.id}>{p.name} · {p.role} · {p.team}</SelectItem>)}</SelectContent></Select><div className="mt-6 flex items-end justify-between"><div><p className="text-sm text-muted-foreground">ZCPV / 90</p><p className="mt-1 text-5xl font-semibold tracking-[-.06em] text-lime-300">+{total.toFixed(2)}</p></div><span className="rounded-full border border-white/10 px-3 py-1 text-xs text-muted-foreground">{player.minutes} min</span></div><div className="mt-7 space-y-4"><CompareBar label="Action" value={player.action} max={1.2} cyan/><CompareBar label="Spatial" value={player.spatial} max={1.2}/><CompareBar label="Combined" value={total} max={2.4}/></div><div className="mt-6 grid grid-cols-4 gap-1.5">{player.zones.map((z,i)=><span key={i} className="aspect-square rounded-sm" style={{background:`rgba(190,242,100,${.08+Math.min(.75,z*3)})`}} title={`${zoneNames[i]}: ${z}`}/>)}</div></div>; }
function CompareBar({label,value,max,cyan=false}:{label:string;value:number;max:number;cyan?:boolean}) { return <div><div className="mb-1.5 flex justify-between text-sm"><span>{label}</span><b>{value>=0?'+':''}{value.toFixed(2)}</b></div><div className="h-2 rounded-full bg-white/8"><div className={`h-full rounded-full ${value<0?'bg-rose-400':cyan?'bg-cyan-400':'bg-lime-300'}`} style={{width:`${Math.min(100,Math.abs(value)/max*100)}%`}}/></div></div>; }

function MethodView() { const max=Math.max(...benchmarks.map(b=>b.naive)); return <div className="grid gap-5 xl:grid-cols-[1fr_.85fr]"><section className="panel p-5 sm:p-7"><p className="eyebrow">Pipeline</p><h2 className="mt-1 text-2xl font-semibold">From tracking frames to player value</h2><div className="mt-7 space-y-3"><MethodStep n="01" icon={<BarChart3/>} title="Learn zone value" text="Estimate possession transition and terminal scoring probabilities, then solve the xT fixed point."/><MethodStep n="02" icon={<Activity/>} title="Value actions" text="Score passes and carries by zone delta, shots by xG, take-ons by retained possession risk, and regains by prevented threat."/><MethodStep n="03" icon={<ScanLine/>} title="Remove one player" text="Compute full and leave-one-out pitch control, reduce cell deltas into 12 tactical zones, and weight them by threat."/><MethodStep n="04" icon={<Check/>} title="Normalize and expose" text="Aggregate action and spatial components per player, normalize per 90, and preserve the decomposition."/></div></section><section className="panel p-5 sm:p-7"><div className="flex items-start justify-between"><div><p className="eyebrow">Measured synthetic benchmark</p><h2 className="mt-1 text-2xl font-semibold">CPU baseline vs. reduced form</h2></div><Zap className="text-lime-300"/></div><p className="mt-3 text-sm leading-relaxed text-muted-foreground">Measured on this build host at 32 × 21 cells. CUDA was unavailable, so no GPU speedup is claimed; the runnable CUDA kernel is included for measurement on NVIDIA hardware.</p><div className="mt-7 space-y-4">{benchmarks.map(b=><div key={b.frames} className="grid grid-cols-[60px_1fr_50px] items-center gap-3"><span className="text-xs text-muted-foreground">{b.frames}f</span><div className="space-y-1"><div className="h-2 rounded bg-cyan-400" style={{width:`${Math.max(3,b.naive/max*100)}%`}}/><div className="h-2 rounded bg-lime-300" style={{width:`${Math.max(3,b.optimized/max*100)}%`}}/></div><b className="text-right text-xs">{(b.naive/b.optimized).toFixed(1)}×</b></div>)}</div><div className="mt-6 flex gap-4 text-xs text-muted-foreground"><span><i className="mr-1 inline-block size-2 rounded-full bg-cyan-400"/>Naive CPU</span><span><i className="mr-1 inline-block size-2 rounded-full bg-lime-300"/>Reduced CPU</span></div></section></div>; }

function MethodStep({n,icon,title,text}:{n:string;icon:React.ReactNode;title:string;text:string}) { return <div className="flex gap-4 rounded-xl border border-white/8 bg-white/[.025] p-4"><span className="font-mono text-xs text-lime-300">{n}</span><span className="mt-0.5 text-cyan-300 [&>svg]:size-4">{icon}</span><div><h3 className="font-semibold">{title}</h3><p className="mt-1 text-sm leading-relaxed text-muted-foreground">{text}</p></div></div>; }
function MiniMetric({icon,label,value}:{icon:React.ReactNode;label:string;value:string}) { return <div className="hidden items-center gap-3 rounded-xl border border-white/10 bg-white/[.035] px-3 py-2.5 sm:flex"><span className="text-lime-300">{icon}</span><div><p className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</p><p className="text-xs font-medium">{value}</p></div></div>; }
