// One thread per (frame, cell); only 22 * 12 aggregates are written per frame.
// Compile directly or at runtime with CuPy RawKernel. No full LOO surfaces exist.
extern "C" __global__ void zcpv_leave_one_out(
    const float* positions, const float* velocities, const int* teams,
    const float* zone_values, float* output, int frames, int grid_x, int grid_y) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  int cells = grid_x * grid_y;
  if (task >= frames * cells) return;
  int frame = task / cells, cell = task % cells;
  int ix = cell % grid_x, iy = cell / grid_x;
  float x = 105.0f * ix / (grid_x - 1), y = 68.0f * iy / (grid_y - 1);
  float weight[22], team_sum[2] = {0.0f, 0.0f};
  #pragma unroll
  for (int p = 0; p < 22; ++p) {
    int base = (frame * 22 + p) * 2;
    float px = positions[base] + .315f * velocities[base];
    float py = positions[base + 1] + .315f * velocities[base + 1];
    float dx = x - px, dy = y - py;
    float tti = .7f + sqrtf(dx * dx + dy * dy) / 7.0f;
    weight[p] = expf(-1.35f * tti);
    team_sum[teams[p]] += weight[p];
  }
  float total = fmaxf(team_sum[0] + team_sum[1], 1e-12f);
  int zone = min(3, ix * 4 / grid_x) * 3 + min(2, iy * 3 / grid_y);
  #pragma unroll
  for (int p = 0; p < 22; ++p) {
    int own = teams[p];
    float full = team_sum[own] / total;
    float without = (team_sum[own] - weight[p]) / fmaxf(total - weight[p], 1e-12f);
    float delta = fmaxf(0.0f, full - without) * zone_values[zone];
    atomicAdd(&output[(frame * 22 + p) * 12 + zone], delta);
  }
}
