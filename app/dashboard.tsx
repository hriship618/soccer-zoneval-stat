import pivotRankings from './pivot-rankings.generated';

export function Dashboard() {
  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <p className="kicker">Player Impact via Outcomes and Tracking</p>
          <h1>PIVOT</h1>
        </div>
        <div className="dataset">
          <b>{pivotRankings.players.length}</b> players · <b>{pivotRankings.matches}</b> matches
        </div>
      </header>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="rank">Rank</th>
              <th>Player</th>
              <th>Team</th>
              <th className="numeric">Minutes</th>
              <th className="numeric">Samples</th>
              <th className="numeric rating-head">PIVOT</th>
            </tr>
          </thead>
          <tbody>
            {pivotRankings.players.map((player) => (
              <tr key={player.player_id}>
                <td className="rank">{player.rank}</td>
                <td className="player">
                  <b>{player.name}</b>
                  <span>{player.role}</span>
                </td>
                <td>{player.team}</td>
                <td className="numeric tabular">{player.minutes.toFixed(1)}</td>
                <td className="numeric tabular">{player.event_samples.toLocaleString()}</td>
                <td className="numeric rating tabular">{player.pivot_rating.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
