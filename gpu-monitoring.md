# GPU Monitoring Follow-up

This note captures the current GPU monitoring state for the tracked Lunar Lake machine and the follow-up work we should do before claiming a complete Intel iGPU telemetry story.

Status labels in this file:

- `locally validated`: directly checked on this repo's tracked machine
- `inference`: a reasonable interpretation of the observed behavior, but not yet fully proven

## Scope

- machine: [`00-setup/systems/lunarlake-ultra7-258v-32gb.md`](/home/lhl/github/lhl/intel-inference/00-setup/systems/lunarlake-ultra7-258v-32gb.md)
- date validated: 2026-03-24
- OS: Arch Linux
- kernel driver: `xe`
- GPU: `Intel Arc Graphics 130V / 140V`

This is machine-specific. Do not generalize it to every Intel Xe-family iGPU without rechecking.

## Current answer

### What is valid right now

`locally validated`

We do have a usable low-level monitoring surface through `xe` sysfs:

- `/sys/class/drm/card0/device/tile0/gt0/freq0/`
- `/sys/class/drm/card0/device/tile0/gt1/freq0/`
- `/sys/class/drm/card0/device/tile0/gt0/gtidle/`
- `/sys/class/drm/card0/device/tile0/gt1/gtidle/`
- `/sys/class/drm/card0/device/tile0/gt0/freq0/throttle/`
- `/sys/class/drm/card0/device/tile0/gt1/freq0/throttle/`

These nodes currently expose:

- requested/current GT frequency via `cur_freq`
- active GT frequency via `act_freq`
- min/max and RP frequency bounds
- GT idle state via `idle_status`
- GT idle residency via `idle_residency_ms`
- throttle status and throttle reasons

During a real Torch XPU FP16 matmul loop, these counters changed in a way that tracked GPU activity:

- idle `gt0`: `cur_freq=1533`, `act_freq=0`, `idle_status=gt-c6`
- under load `gt0`: `cur_freq=1950`, `act_freq=1350..1950`, `idle_status=gt-c0`
- after load `gt0`: dropped back through lower `cur_freq` values with `act_freq=0` and `gt-c6`
- `gt1` stayed idle for that compute workload

That is enough to say we have valid limited monitoring for:

- GT frequency behavior
- GT active vs idle state
- GT idle residency
- GT throttle flags

### What is not valid right now

`locally validated`

The higher-level tools we would normally prefer are not usable on this machine today:

- `xpu-smi discovery` returns `No device discovered`
- `xpu-smi stats -d 0 -j` fails with `Level Zero Initialization Error`
- `intel_gpu_top -L` lists the device, but live sampling fails with:
  - `Failed to detect engines! (No such file or directory)`
- the tracked telemetry capture already records `intel_gpu_top` as `unsupported`

We also do not currently have a validated temperature or power path for the iGPU:

- `/sys/class/drm/card0/device/hwmon` is absent here
- `/sys/class/drm/card0/device/drm/card0/metrics/` exists but is empty here
- `sensors` did not expose a GPU-specific temp or power node on this machine

That means we should not currently claim valid monitoring for:

- engine utilization percentages
- package or GT power draw
- GPU temperature
- VRAM or local-memory telemetry

## Recommended current stance in repo docs

Use this wording until the monitoring story improves:

- GPU functionality is validated on this machine.
- GPU telemetry is only partially validated on this machine.
- `xpu-smi` is not authoritative here.
- `intel_gpu_top` is not usable here on the current `xe` stack.
- `xe` sysfs GT frequency, idle, and throttle state are the current best low-level monitoring path.

Do not use stronger language than that yet.

## Minimal commands that work now

### Quick idle-state check

```bash
for f in \
  /sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq \
  /sys/class/drm/card0/device/tile0/gt0/freq0/act_freq \
  /sys/class/drm/card0/device/tile0/gt0/gtidle/idle_status \
  /sys/class/drm/card0/device/tile0/gt0/gtidle/idle_residency_ms \
  /sys/class/drm/card0/device/tile0/gt1/freq0/cur_freq \
  /sys/class/drm/card0/device/tile0/gt1/freq0/act_freq \
  /sys/class/drm/card0/device/tile0/gt1/gtidle/idle_status \
  /sys/class/drm/card0/device/tile0/gt1/gtidle/idle_residency_ms
do
  printf '%s=%s\n' "$f" "$(cat "$f")"
done
```

### Quick throttle-state check

```bash
for f in \
  /sys/class/drm/card0/device/tile0/gt0/freq0/throttle/status \
  /sys/class/drm/card0/device/tile0/gt0/freq0/throttle/reasons \
  /sys/class/drm/card0/device/tile0/gt1/freq0/throttle/status \
  /sys/class/drm/card0/device/tile0/gt1/freq0/throttle/reasons
do
  printf '%s=%s\n' "$f" "$(cat "$f")"
done
```

### Confirm that `intel_gpu_top` is still broken

```bash
intel_gpu_top -L
intel_gpu_top -J -s 200 -n 5 -d pci:vendor=8086,device=64A0,card=0
```

### Confirm that `xpu-smi` is still not authoritative

```bash
xpu-smi discovery
xpu-smi stats -d 0 -j
```

## Follow-up work

1. Add a small repo helper that samples the `xe` sysfs GT nodes to CSV or JSON while a benchmark command runs.
2. Test a media workload to see whether `gt1`, `vcs`, or `vecs` expose any activity pattern distinct from the compute-heavy `gt0` path.
3. Recheck `intel_gpu_top` after future kernel, Mesa, or `intel-gpu-tools` updates.
4. Recheck `xpu-smi` after future Level Zero and userspace stack updates.
5. Look again for a validated iGPU power or temperature surface under `xe`, `hwmon`, or another kernel-exposed metrics path.
6. If a stable monitor appears, wire it into [`01-hardware/run-suite.sh`](/home/lhl/github/lhl/intel-inference/01-hardware/run-suite.sh).

## Open questions

- `inference`: `intel_gpu_top` may still depend on an engine/PMU path that is not exposed in a usable way on this `xe` setup.
- `inference`: `xpu-smi` failure may be a userspace Level Zero support gap for this iGPU path rather than proof that the GPU stack itself is unhealthy.

These are plausible readings of the failure modes, but we should keep them labeled as inference until we verify them against upstream docs or future tool updates.
