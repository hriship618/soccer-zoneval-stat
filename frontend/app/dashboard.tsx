import pivotRankings from './pivot-rankings.generated';

type RankingPlayer =
  | (typeof pivotRankings.outfield_players)[number]
  | (typeof pivotRankings.goalkeepers)[number];

function RankingTable({ title, players }: { title: string; players: readonly RankingPlayer[] }) {
  return (
    <section className="ranking-section">
      <h2>{title}</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="rank">Rank</th>
              <th>Player</th>
              <th>Team</th>
              <th className="numeric">Matches</th>
              <th className="numeric">Minutes</th>
              <th className="numeric rating-head">PIVOT</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => (
              <tr key={player.player_id}>
                <td className="rank">{player.rank}</td>
                <td className="player"><b>{player.name}</b></td>
                <td>{player.team}</td>
                <td className="numeric tabular">{player.matches}</td>
                <td className="numeric tabular">{player.minutes.toFixed(1)}</td>
                <td className="numeric rating tabular">{player.pivot_rating.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function Dashboard() {
  const playerCount = pivotRankings.outfield_players.length + pivotRankings.goalkeepers.length;
  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <p className="kicker">Player Impact via Outcomes and Tracking</p>
          <h1>PIVOT</h1>
        </div>
        <div className="dataset">
          <b>{playerCount}</b> players · <b>{pivotRankings.matches}</b> matches
        </div>
      </header>
      <RankingTable title="Outfield" players={pivotRankings.outfield_players} />
      <RankingTable title="Goalkeepers" players={pivotRankings.goalkeepers} />
    </main>
  );
}
