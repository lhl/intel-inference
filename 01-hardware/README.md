# 01-hardware

This directory is for low-level hardware characterization before framework or model-level testing.

The point of this stage is to establish ceilings and bottlenecks:

- host memory bandwidth
- GPU-accessible memory bandwidth
- NPU-relevant bandwidth or transfer limits if measurable
- GEMM or FLOPs-style compute throughput
- thermal, clock, and power behavior during sustained load

## What belongs here

- raw memory-bandwidth microbenchmarks
- raw compute microbenchmarks
- device-transfer tests
- telemetry capture
- notes on which tools are credible and which are misleading
- saved raw outputs and summarized results

## Exit criteria

We should not treat framework-level numbers as meaningful until we have:

1. at least one repeatable system-memory bandwidth measurement
2. at least one repeatable GPU-focused bandwidth measurement
3. a basic compute microbench path for the GPU
4. a record of clocks, power mode, and thermal conditions during those runs
5. clear notes on what is measured and what is only inferred

## Expected work products

Likely files or scripts to add here later:

- `run-mbw.sh`
- `run-gemm.sh`
- `collect-telemetry.sh`
- `results/`
- `notes.md`

## Related docs

- [RESEARCH-hardware.md](/home/lhl/github/lhl/intel-inference/RESEARCH-hardware.md)
- [TESTING.md](/home/lhl/github/lhl/intel-inference/TESTING.md)
