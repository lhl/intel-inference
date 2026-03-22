# Hardware Research

Last updated: 2026-03-22

This note captures the current public hardware picture for one concrete Lunar Lake mobile SKU and one launched Panther Lake mobile SKU, with an emphasis on inference-relevant specs: clocks, published AI TOPS, visible precision support, and the places where Intel does not publish a clean FLOPS number.

## Scope and caveats

- The concrete comparison below uses `Intel Core Ultra 7 258V` as the Lunar Lake reference point and `Intel Core Ultra X7 358H` as the Panther Lake reference point.
- Intel publishes most client AI marketing numbers as peak `INT8 TOPS`, split across GPU and NPU, or as family-level "platform TOPS".
- Intel ARK pages for these integrated mobile parts do not currently publish a GPU `FP32 TFLOPS` figure.
- Precision support is stack-dependent. Where ARK does not publish a precision matrix directly, this document uses Intel's OpenVINO and oneDNN docs as the software-visible precision signal.
- Panther Lake family marketing currently says "up to 180 platform TOPS", but the `358H` ARK page does not currently list a SKU-level overall platform TOPS field.

## Concrete SKU comparison

| Field | Lunar Lake: Core Ultra 7 258V | Panther Lake: Core Ultra X7 358H | Notes |
|---|---:|---:|---|
| Launch | Q3'24 | Q1'26 | ARK launch quarter |
| CPU lithography | TSMC N3B | Intel 18A | ARK |
| CPU cores / threads | 8 / 8 | 16 / 16 | `258V`: 4P + 4 LP-E. `358H`: 4P + 8E + 4 LP-E |
| CPU max turbo | 4.8 GHz | 4.8 GHz | ARK |
| Base power / max turbo power | 17 W / 37 W | 25 W / 80 W | ARK |
| Max memory | 32 GB | 96 GB | ARK |
| Memory type | LPDDR5X-8533 | LPDDR5X-9600 | ARK |
| GPU name | Arc 140V | Arc B390 | ARK |
| GPU microarchitecture | Xe2-LPG | Xe3-class Arc branding, exact public oneDNN row not yet published | ARK plus Intel docs gap |
| GPU Xe-cores | 8 | 12 | ARK |
| GPU max clock | 1.95 GHz | 2.5 GHz | ARK |
| GPU peak TOPS (INT8) | 64 | 122 | ARK |
| Advertised GPU TFLOPS | Not listed on ARK | Not listed on ARK | Intel does publish TFLOPS for some discrete Arc SKUs, but not these iGPU ARK pages |
| NPU peak TOPS (INT8) | 47 | 50 | ARK |
| Published overall platform TOPS | 115 | Not listed on `358H` ARK; Panther Lake family advertises up to 180 | `258V` ARK is SKU-specific; Panther value is family-level marketing |
| GPU frameworks listed on ARK | OpenVINO, WindowsML, DirectML, ONNX RT, WebGPU, WebNN | OpenVINO, WindowsML, DirectML, ONNX RT, WebGPU, WebNN | ARK |
| NPU frameworks listed on ARK | OpenVINO, WindowsML, DirectML, ONNX RT, WebNN | OpenVINO, WindowsML, WebNN, ONNX RT | ARK |

## Reference dGPU comparison

| Field | Arc A580 | Arc B580 | Notes |
|---|---:|---:|---|
| Launch | Q4'23 | Q4'24 | ARK |
| Microarchitecture | Xe-HPG | Xe2 | ARK |
| Lithography | TSMC N6 | TSMC N5 | ARK |
| Xe-cores | 24 | 20 | ARK |
| XMX engines | 384 | 160 | ARK |
| Vector engines | 384 | 160 | ARK |
| Graphics clock | 1700 MHz | 2670 MHz | ARK |
| Published GPU peak TOPS (INT8) | 197 | 233 | ARK |
| VRAM | 8 GB GDDR6 | 12 GB GDDR6 | ARK |
| VRAM interface | 256-bit | 192-bit | ARK |
| VRAM speed | 16 Gbps | 19 Gbps | ARK |
| Published memory bandwidth | 512 GB/s | 456 GB/s | ARK |
| TBP | 185 W | 190 W | ARK |
| PCIe | Up to PCIe 4.0 x16 | PCIe 4.0 x8 | ARK |

## Implied peak tensor clock

This table back-solves the clock that would be needed to exactly hit the published `INT8 TOPS` number, using the per-engine tensor assumptions currently used in this note.

| Device | Published INT8 TOPS | Assumed tensor structure | Public clock field | Implied tensor clock | Read |
|---|---:|---|---:|---:|---|
| Core Ultra 7 258V | 64 | `8 Xe-cores * 8 XMX/Xe-core * 512 INT8 ops/XMX/clock` | `1.95 GHz` max GPU clock | `1.953 GHz` | Essentially exact match |
| Core Ultra X7 358H | 122 | `12 Xe-cores * 8 XMX/Xe-core * 512 INT8 ops/XMX/clock` | `2.50 GHz` max GPU clock | `2.482 GHz` | Very close; within rounding noise |
| Arc A580 | 197 | `24 Xe-cores * 16 XMX/Xe-core * 256 INT8 ops/XMX/clock` | `1.70 GHz` graphics clock | `2.004 GHz` | Strong evidence Intel's TOPS figure uses a higher effective peak clock than ARK's listed graphics clock |
| Arc B580 | 233 | `20 Xe-cores * 8 XMX/Xe-core * 512 INT8 ops/XMX/clock` | `2.67 GHz` graphics clock | `2.844 GHz` | Likely peak/boost/tensor-side clock basis rather than the public graphics clock field |

The pattern is useful:

- `258V` and `358H` line up well with the public clock fields.
- `A580` and `B580` do not.
- That makes integrated client ARK pages more trustworthy for direct clock-based reconstruction than current discrete Arc ARK pages.

## Bandwidth reference

For local inference, bandwidth is often more predictive than peak TOPS:

- iGPU and NPU traffic runs over shared system memory bandwidth.
- dGPU compute runs primarily from local GDDR bandwidth.
- A desktop dGPU system also has host DDR5 bandwidth, but that is not the same thing as the GPU's device-local bandwidth.

### Formula

```text
Theoretical memory bandwidth (GB/s) ~= MT/s * BusWidthBits / 8 / 1000
```

### Device-local or shared-memory ceilings

| Device | Memory type | Width | Rate | Theoretical bandwidth | Notes |
|---|---|---:|---:|---:|---|
| Core Ultra 7 258V iGPU/NPU/shared system memory | LPDDR5X | 128-bit | 8533 MT/s | `136.5 GB/s` | Derived from ARK max memory type and standard 2 x 64-bit package interface |
| Core Ultra X7 358H iGPU/NPU/shared system memory | LPDDR5X | 128-bit | 9600 MT/s | `153.6 GB/s` | Derived from ARK max memory type and standard dual-channel 128-bit aggregate width |
| Arc A580 local VRAM | GDDR6 | 256-bit | 16 Gbps | `512 GB/s` | Matches ARK published bandwidth |
| Arc B580 local VRAM | GDDR6 | 192-bit | 19 Gbps | `456 GB/s` | Matches ARK published bandwidth |

### Common host DDR5 reference points

These matter mostly for CPU-side preprocessing, host staging, and PCIe-fed transfers to dGPUs:

| Memory configuration | Aggregate width | Data rate | Theoretical bandwidth |
|---|---:|---:|---:|
| DDR5-5600 dual channel | 128-bit | 5600 MT/s | `89.6 GB/s` |
| DDR5-6000 dual channel | 128-bit | 6000 MT/s | `96.0 GB/s` |
| DDR5-6400 dual channel | 128-bit | 6400 MT/s | `102.4 GB/s` |

## Precision support notes

### CPU

| Area | Lunar Lake 258V | Panther Lake 358H | Research note |
|---|---|---|---|
| ISA signal on ARK | `AVX2`, Intel DL Boost | `AVX2`, Intel DL Boost | Neither ARK page exposes AMX on these mobile parts |
| Published CPU AI peak metric | No standalone CPU TOPS value on ARK | No standalone CPU TOPS value on ARK | Intel rolls client AI marketing into platform/GPU/NPU headlines |
| Practical inference note | CPU remains relevant for orchestration, sampling, tokenization, fallback ops, and some OpenVINO / oneDNN CPU paths | Same | Good control-plane engine; not the headline AI accelerator |

### GPU

| Area | Lunar Lake Xe2-LPG | Panther Lake Arc B390 | Research note |
|---|---|---|---|
| Published peak AI metric | `64 INT8 TOPS` | `122 INT8 TOPS` | ARK |
| Published floating-point throughput | Not listed on ARK | Not listed on ARK | This is the main documentation gap if we want a clean TFLOPS comparison |
| oneDNN performant data types | `f64`, `f32`, `bf16`, `f16`, `s8`, `u8` for `Xe2-LPG` | No Panther/Xe3 row published yet in the oneDNN 2025.2 table | Intel oneDNN docs are better than ARK here, but currently much clearer for Xe2 than Xe3 |
| OpenVINO GPU plugin data types | `f32`, `f16`, `u8`, `i8`, `u1` internal primitive support | Same plugin model, but Panther-specific public precision table is still thin | OpenVINO docs describe plugin-visible types, not a full per-SKU peak-math matrix |

### NPU

| Area | Lunar Lake NPU | Panther Lake NPU | Research note |
|---|---|---|---|
| Published peak AI metric | `47 INT8 TOPS` | `50 INT8 TOPS` | ARK |
| OpenVINO internal inference data types | `FP32`, `FP16`, quantized `U8` with mixed `FP16-INT8` models; hardware compute precision is `FP16` | Same public NPU device model today | OpenVINO is the clearest public source for NPU-visible precision |
| LLM export constraints in OpenVINO GenAI | Symmetric `INT4` or `NF4` weight export is required for current NPU LLM flows | Same direction, "Series 2 and beyond" wording covers Panther Lake too | This matters more than raw NPU TOPS for real local-LLM work |
| LLM runtime caveat | NPU LLM path still carries NPU-specific config and static-shape roots, with dynamic prompt support added in OpenVINO 2025.3 | Same | The software constraints are at least as important as the peak TOPS number |

## Family-level marketing numbers

| Family | Intel public claim | Source note |
|---|---|---|
| Lunar Lake | Up to `120 platform TOPS` family headline; concrete `258V` ARK page lists `115 overall peak TOPS (INT8)` | Intel family marketing and SKU ARK line up directionally but are not the same number |
| Panther Lake | Up to `180 platform TOPS` family headline | Intel infographic / newsroom material; no matching SKU-level overall TOPS field on `358H` ARK at the time of writing |

## Derived throughput estimates

Yes, we can derive a useful estimate from published clock rates and Intel's Xe core structure, but only if we separate:

- `documented`: values Intel publishes directly on ARK
- `derived`: values implied by other official Intel Xe2 disclosures
- `inferred`: values that fit Panther Lake's published numbers but are not yet backed by a public Xe3 per-core throughput table

### Derivation basis

For Intel's launched discrete `Xe2-HPG` part `Arc Pro B60`, Intel publishes all of the following on one official page:

- `20 Xe-cores`
- `160 XMX engines`
- `160 vector engines`
- `2400 MHz graphics clock`
- `12.28 TFLOPS (FP32)`
- `197 TOPS (INT8)`

Those figures are consistent with the following per-engine rates for Xe2:

- `8 vector engines per Xe-core`
- `8 XMX engines per Xe-core`
- about `32 FP32 ops / vector engine / clock`
- about `512 INT8 ops / XMX / clock`

That yields the following simple formulas:

```text
FP32 TFLOPS ~= XeCores * VectorEnginesPerXeCore * FP32OpsPerVectorEnginePerClock * ClockGHz / 1000

INT8 TOPS ~= XeCores * XMXPerXeCore * INT8OpsPerXMXPerClock * ClockGHz / 1000
```

If we keep the same Xe2 assumptions for FP16 and BF16 tensor math, the matching matrix rates are:

- about `256 FP16/BF16 ops / XMX / clock`
- about `1024 INT4 ops / XMX / clock`

### Estimated GPU throughput table

| SKU | Status | Assumptions | Derived FP32 TFLOPS | Derived vector FP16 TFLOPS | Derived XMX FP16/BF16 TFLOPS | Derived XMX INT8 TOPS | Derived XMX INT4 TOPS |
|---|---|---|---:|---:|---:|---:|---:|
| Core Ultra 7 258V | Derived, fairly strong | `8 Xe-cores`, `1.95 GHz`, `8 VE/Xe-core`, `8 XMX/Xe-core`, Xe2 per-engine rates inferred from Arc Pro B60 | `3.99` | `7.99` | `31.95` | `63.90` | `127.80` |
| Core Ultra X7 358H | Inferred, medium confidence | `12 Xe-cores`, `2.5 GHz`, and Panther keeping the same per-engine rates as Xe2 | `7.68` | `15.36` | `61.44` | `122.88` | `245.76` |

### Why the Lunar Lake estimate is relatively solid

- Intel's oneAPI Xe architecture table explicitly gives Lunar Lake `Xe2-LPG` as `8 Xe-cores` and `64 vector engines`, which already implies `8 vector engines per Xe-core`.
- The `258V` ARK page gives `1.95 GHz` max GPU clock and `64 INT8 TOPS`.
- Plugging `8 Xe-cores * 8 XMX/Xe-core * 512 INT8 ops/clock * 1.95 GHz` gives `63.90 TOPS`, which rounds to Intel's published `64 TOPS`.

So for Lunar Lake, the back-solved estimate is internally consistent with Intel's published ARK number.

### Why the Panther Lake estimate is weaker

- The `358H` ARK page gives `12 Xe-cores`, `2.5 GHz`, and `122 INT8 TOPS`.
- If Panther's integrated `Arc B390` keeps the same `8 XMX per Xe-core` and `512 INT8 ops/XMX/clock` structure, the result is `122.88 TOPS`, which matches the published `122 TOPS` after rounding.
- That makes the estimate plausible, but Intel has not yet published the same kind of Xe3 per-core throughput table that would let us call `7.68 TFLOPS FP32` a directly documented number.

### Practical takeaway

- For `Lunar Lake 258V`, a reasonable working estimate is:
  - `~4.0 TFLOPS FP32`
  - `~32 TFLOPS` matrix `FP16/BF16`
  - `~64 TOPS INT8`
- For `Panther Lake 358H`, a reasonable best-fit estimate is:
  - `~7.7 TFLOPS FP32`
  - `~61.4 TFLOPS` matrix `FP16/BF16`
  - `~122 TOPS INT8`

These are theoretical peak compute figures. They do not imply equivalent LLM decode performance, which is usually limited much more by memory bandwidth, cache behavior, and runtime support than by raw peak math.

## Derived dGPU notes

### Arc A580

For `Xe-HPG` Intel has published a more explicit per-engine throughput table than it has for current Xe2/Xe3 client parts:

- `16 FP32 FLOPs / vector engine / clock`
- `32 FP16 FLOPs / vector engine / clock`
- `128 FP16/BF16 ops / XMX / clock`
- `256 INT8 ops / XMX / clock`

Using the listed `1700 MHz` graphics clock for `A580` gives:

- `~10.44 TFLOPS FP32`
- `~20.89 TFLOPS` vector `FP16`
- `~83.56 TOPS` XMX `FP16/BF16`
- `~167.12 TOPS` XMX `INT8`

Important caveat:

- Intel's published `A580` figure is `197 INT8 TOPS`, which does not line up with the older Xe-HPG per-XMX formula at `1700 MHz`.
- Back-solving from `197 TOPS` implies an effective tensor-side clock of about `2.00 GHz`.
- Intel's A580 launch material also pointed to partner cards running at `2000 MHz`, which matches the published `197 TOPS` almost exactly.
- So for `A580`, the safest reading is that the public TOPS number is official, while any TFLOPS derived from the listed graphics clock should be treated as a clock-based estimate, not as an Intel-published peak-float number.

### Arc B580

Using the same Xe2 per-engine behavior inferred earlier from `Arc Pro B60`, the `B580` listed `2670 MHz` clock implies:

- `~13.67 TFLOPS FP32`
- `~27.34 TFLOPS` vector `FP16`
- `~109.36 TOPS` XMX `FP16/BF16`
- `~218.73 TOPS` XMX `INT8`

Important caveat:

- Intel's published `B580` figure is `233 INT8 TOPS`, which is somewhat above the simple `2670 MHz` clock-based estimate.
- That means either the relevant peak tensor clock is higher than the listed graphics clock, or Intel is measuring peak throughput using a slightly different clock basis than the public ARK `Graphics Clock` field.
- So `B580` is useful as a reference Xe2 dGPU, but its public clock field should not be treated as a perfect key for reconstructing every published peak number.

## External sanity checks

Two Reddit posts by the repo author are useful as secondary Lunar Lake corroboration:

- `Testing llama.cpp with Intel's Xe2 iGPU (Core Ultra 7 258V w/ Arc Graphics 140V)`
- `llama.cpp Compute and Memory Bandwidth Efficiency w/ Different Devices/Backends`

What they corroborate for `Core Ultra 7 258V`:

- `32 GB LPDDR5-8533` implies a `136.5 GB/s` theoretical shared-memory ceiling.
- `Arc 140V` belongs in the `~32 FP16 TFLOPS` theoretical class for matrix math.
- In real llama.cpp usage, prompt processing tracks compute much more closely than token generation does.
- Token generation remains strongly bandwidth-limited, which is consistent with the large gap between theoretical TOPS/TFLOPS and observed decode throughput on shared-memory client parts.

These are not primary hardware-spec sources, but they are a useful reality check that the table math here is pointing in the right direction for Lunar Lake.

## Working conclusions

- Lunar Lake already crossed the point where the GPU is the dominant published AI engine for local inference math on the client part: `64 GPU TOPS` versus `47 NPU TOPS` on the `258V`.
- Panther Lake scales that pattern further: `122 GPU TOPS` and `50 NPU TOPS` on the `358H`, with the headline family claim moving to `up to 180 platform TOPS`.
- Shared-memory client parts are still in a very different bandwidth class from dGPUs:
  - `258V`: about `136.5 GB/s`
  - `358H`: about `153.6 GB/s`
  - `A580`: `512 GB/s`
  - `B580`: `456 GB/s`
- That bandwidth gap is a big part of why raw client GPU/NPU TOPS does not translate cleanly into dGPU-class LLM behavior.
- Intel's public mobile-client documentation is much cleaner on `TOPS` than on `TFLOPS`. If this repo needs a FLOPS table, we will likely need either:
  - a later Intel product page that explicitly publishes FP32/FP16 throughput for these integrated GPUs, or
  - a clearly labeled derived-estimate section based on architecture docs, not ARK.
- For inference planning, the more actionable precision signal is not the marketing TOPS number but the stack-visible one:
  - Xe2-LPG GPU: good public signal for `f32` / `bf16` / `f16` / `s8` / `u8`
  - NPU: public signal for `FP16` compute with quantized flows, and current OpenVINO NPU LLM export expectations centered on `INT4` / `NF4`
  - Panther/Xe3 GPU: public per-SKU TOPS is already available, but the software precision matrix is not yet as explicit in Intel's docs as it is for Xe2-LPG

## Sources

- Intel ARK: Core Ultra 7 258V
  - https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html
- Intel ARK: Core Ultra X7 358H
  - https://www.intel.com/content/www/us/en/products/sku/245527/intel-core-ultra-x7-processor-358h-18m-cache-up-to-4-80-ghz/specifications.html
- Intel Newsroom: Panther Lake by the numbers
  - https://newsroom.intel.com/client-computing/introducing-panther-lake-by-the-numbers
- Intel infographic PDF: Panther Lake up to 180 platform TOPS
  - https://download.intel.com/newsroom/2025/client-computing/itt-panther-lake-Infographic.pdf
- Intel oneAPI optimization guide: Xe GPU architecture table
  - https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- Intel oneDNN data types guide
  - https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-2/data-types-001.html
- Intel ARK: Arc Pro B60 Graphics
  - https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html
- Intel ARK: Arc A580 Graphics
  - https://www.intel.com/content/www/us/en/products/sku/227961/intel-arc-a580-graphics/specifications.html
- Intel ARK: Arc B580 Graphics
  - https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html
- OpenVINO GPU device docs
  - https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html
- OpenVINO NPU device docs
  - https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO GenAI on NPU
  - https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- Reddit sanity check: Testing llama.cpp with Intel's Xe2 iGPU
  - https://www.reddit.com/r/LocalLLaMA/comments/1gheslj/testing_llamacpp_with_intels_xe2_igpu_core_ultra/
- Reddit sanity check: llama.cpp compute and memory bandwidth efficiency
  - https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/
