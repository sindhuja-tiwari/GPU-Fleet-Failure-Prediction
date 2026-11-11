# Loading real fleet data

`gpufleet ingest` turns NVIDIA [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter)
output into the same `telemetry.parquet` the simulator writes, so every later stage runs
unchanged on a real fleet.

## From Prometheus

If dcgm-exporter is already scraped by Prometheus, pull history with a range query and
save the JSON response. Keep the step equal to your scrape interval.

```bash
curl -G "$PROM/api/v1/query_range" \
  --data-urlencode 'query={__name__=~"DCGM_FI_DEV_.*"}' \
  --data-urlencode "start=2026-03-01T00:00:00Z" \
  --data-urlencode "end=2026-03-02T00:00:00Z" \
  --data-urlencode "step=300" > day1.json

gpufleet ingest --prometheus-json day1.json day2.json --out data/real --rack-map racks.json
```

Large ranges hit Prometheus's 11,000-points-per-series limit; split them by day.

## From raw exporter scrapes

Save each scrape to a file named by its UTC time, then point `ingest` at the directory.

```bash
mkdir -p scrapes
while true; do
  curl -s http://gpu-node-17:9400/metrics > "scrapes/$(date -u +%Y-%m-%dT%H-%M-%S).prom"
  sleep 300
done

gpufleet ingest --exporter-dir scrapes --out data/real
```

## Field mapping

| Schema column | DCGM field (first one found wins) |
| --- | --- |
| `gpu_util` | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE` |
| `mem_util` | `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_PROF_DRAM_ACTIVE` |
| `sm_clock_mhz` | `DCGM_FI_DEV_SM_CLOCK` |
| `power_w` | `DCGM_FI_DEV_POWER_USAGE` |
| `temp_gpu_c` | `DCGM_FI_DEV_GPU_TEMP` |
| `temp_mem_c` | `DCGM_FI_DEV_MEMORY_TEMP` |
| `throttle_thermal`, `throttle_power` | bits of `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` or `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (thermal: 0x20, 0x40; power: 0x04, 0x80) |
| `ecc_sbe_total`, `ecc_dbe_total` | `DCGM_FI_DEV_ECC_[SD]BE_AGG_TOTAL`, then `_VOL_TOTAL` |
| `retired_pages_total` | sum of `DCGM_FI_DEV_RETIRED_SBE`, `_RETIRED_DBE`, `_CORRECTABLE_REMAPPED_ROWS`, `_UNCORRECTABLE_REMAPPED_ROWS` |
| `pcie_replay_total` | `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` |
| `nvlink_crc_total` | `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` |
| `xid` | `DCGM_FI_DEV_XID_ERRORS`, turned into events (see below) |
| `node_id` | `Hostname` label, else `instance` without the port |
| `gpu_index`, `gpu_uuid` | `gpu` and `UUID` labels |
| `job_id` | `pod` label by default (`--job-label` to change, `''` for none) |
| `rack_id` | from `--rack-map` (JSON of node to rack), else `unknown` |
| `step_compute_ms` | not in DCGM: left empty; join from training logs for straggler work |

Things to know:

- **Required fields:** utilisation, SM clock, power and GPU temperature. `ingest` stops
  with an error if any is missing.
- **Missing counters are filled with 0, with a warning.** The default dcgm-exporter config
  leaves out several error counters (ECC, NVLink CRC). Turn them on in its counters CSV,
  otherwise the failure models are blind to memory and NVLink failures.
- **Xid is a last-value gauge in DCGM:** it keeps showing the last Xid long after the
  error. The loader keeps the code only on the scrape where it changed.
- `DCGM_FI_*` fields without a mapping are ignored.
