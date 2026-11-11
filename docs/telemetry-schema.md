# Telemetry schema

One row per GPU per scrape (default every 5 minutes). Column names follow
[DCGM exporter](https://github.com/NVIDIA/dcgm-exporter) fields where one exists, so real
exports can be mapped onto this schema.

Counters ending in `_total` are cumulative since the driver last loaded. They restart at
zero when the GPU is replaced (a new `gpu_uuid` in the same `node_id` / `gpu_index` slot)
and after a node reboot or driver upgrade. Use `gpufleet.clean.counter_increments` to get
per-scrape counts; never difference raw totals.

`timestamp` is the node's own clock at scrape time, so it is a few seconds off the 5-minute
grid and differs between nodes. `gpufleet.clean.align_timestamps` snaps it to the grid.

`gpufleet.schema.validate_telemetry` checks types, ranges, missing values and duplicates,
and that counters only go down after a gap of more than 10 minutes (a reboot).

| Column | Type | Unit | DCGM field | Description |
| --- | --- | --- | --- | --- |
| `timestamp` | time | UTC |  | Scrape time |
| `rack_id` | str |  |  | Rack the node sits in |
| `node_id` | str |  |  | Host name |
| `gpu_index` | int |  |  | GPU slot on the node (0-7) |
| `gpu_uuid` | str |  |  | Physical GPU; changes when a GPU is replaced |
| `job_id` | str |  |  | Job running on the node, empty when idle |
| `gpu_util` | float | % | `DCGM_FI_DEV_GPU_UTIL` | SM utilisation |
| `mem_util` | float | % | `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilisation |
| `sm_clock_mhz` | float | MHz | `DCGM_FI_DEV_SM_CLOCK` | SM clock |
| `power_w` | float | W | `DCGM_FI_DEV_POWER_USAGE` | Board power draw |
| `temp_gpu_c` | float | C | `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature |
| `temp_mem_c` | float | C | `DCGM_FI_DEV_MEMORY_TEMP` | HBM temperature (not reported by every GPU model) |
| `throttle_thermal` | bool |  | `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | Clocks limited by temperature |
| `throttle_power` | bool |  | `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | Clocks limited by power cap |
| `ecc_sbe_total` | int | count | `DCGM_FI_DEV_ECC_SBE_AGG_TOTAL` | Corrected (single-bit) ECC errors |
| `ecc_dbe_total` | int | count | `DCGM_FI_DEV_ECC_DBE_AGG_TOTAL` | Uncorrected (double-bit) ECC errors |
| `retired_pages_total` | int | count | `DCGM_FI_DEV_RETIRED_SBE` | Memory pages retired or rows remapped |
| `pcie_replay_total` | int | count | `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | PCIe replay events |
| `nvlink_crc_total` | int | count | `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | NVLink CRC errors (all links) |
| `xid` | int | code | `DCGM_FI_DEV_XID_ERRORS` | Last Xid error in this interval, 0 if none |
| `step_compute_ms` | float | ms |  | Time this GPU spent on its share of a training step; empty when idle |

## Ground-truth tables

These come from the simulator and are never used as model inputs. Their times are on the
exact step grid (no clock skew).

**`failures.parquet`**: one row per GPU that started degrading inside the window.

| Column | Meaning |
| --- | --- |
| `gpu_uuid`, `rack_id`, `node_id`, `gpu_index` | Which GPU |
| `mode` | `memory`, `thermal`, `nvlink`, `power` or `sudden` |
| `fatal_xid` | Xid logged in the GPU's last row |
| `onset_time` | When degradation started (first moment a perfect model could know) |
| `failure_time` | Timestamp of the last row, which carries the fatal Xid |
| `repaired_time` | When the replacement GPU came online (empty if after the window) |
| `observed` | `False` if the GPU was still degrading when the window ended |

**`stragglers.parquet`**: one row per interval where a GPU ran slower than its peers.

| Column | Meaning |
| --- | --- |
| `kind` | `chronic` (always slow, lower clocks), `transient` (host or network hiccup, clocks look normal), `thermal` (throttling from a failing cooler) |
| `slowdown` | Step-time multiplier, e.g. 1.2 = 20% slower |
| `start`, `end` | Interval, end exclusive |

**`inventory.parquet`**: every physical GPU, with install and removal times.

**`maintenance.parquet`**: one row per maintenance window on a node.

| Column | Meaning |
| --- | --- |
| `kind` | `reboot`, `driver_upgrade`, `drain`, or `gpu_swap` (added when a failed GPU is replaced) |
| `start`, `end` | Window, end exclusive |
| `node_offline` | No telemetry from the node during the window |
| `resets_counters` | DCGM counters on every GPU of the node restart at zero after the window |
