# UMBP Embedded DRAM Offloading Status

This document describes the scope, current implementation status, validation
coverage, and remaining work for the UMBP embedded runtime.

## Scope

The embedded runtime is responsible only for single-engine, single-node KV
cache offloading to MORI-managed host DRAM.

The other deployment modes are separate runtime responsibilities:

```text
embedded runtime
└── In-process, single-node, UMBP-managed DRAM offloading

standalone runtime
└── A separate service and DRAM sharing across engines

distributed runtime
└── Cross-node metadata, RDMA, and distributed tier management
```

Standalone services, cross-engine sharing, cross-node access, RDMA, and SSD
tiers are therefore not completion requirements for embedded mode.

Embedded mode currently rejects pipeline parallelism (`PP > 1`). PP stages need
a stage-specific KV ownership map before logical object completeness can be
implemented safely.

## Data Path

Embedded mode registers each rank's GPU KV buffers with MORI and transfers KV
objects between GPU memory and the UMBP DRAM tier.

```text
vLLM GPU KV cache
  └── batch_put_ranges_from_ptr()
      └── UMBP-managed host DRAM
          └── batch_get_ranges_into_ptr()
              └── vLLM GPU KV cache
```

vLLM owns request scheduling, GPU KV blocks, block hashes, and transfer plans.
MORI owns DRAM allocation, object lookup, publication, capacity management, and
eviction.

## Current Features

- [x] GPU-to-DRAM KV store.
- [x] DRAM-to-GPU KV load.
- [x] UMBP object lookup and publication.
- [x] Configurable per-rank DRAM capacity.
- [x] Configurable high and low eviction watermarks.
- [x] Configurable NUMA placement.
- [x] Configurable huge pages and huge-page size.
- [x] Configurable memory prefaulting.
- [x] Configurable shared-memory backing and name.
- [x] Asynchronous transfer execution.
- [x] Free-queue-driven lazy offload with rank-aggregated store events.
- [x] Transfer timeout and per-object failure reporting.
- [x] Request preemption and transfer cancellation lifecycle.
- [x] GPU block pinning until store completion.
- [x] Cache reset.
- [x] Ranged I/O over multi-layer KV objects.
- [x] Partial hash hits and partial physical ranges.
- [x] Non-block-aligned partial-tail plans.
- [x] Hybrid and Mamba boundary-state planning.
- [x] Layer-wise load.
- [x] TP, PP, PCP, and DCP rank-local object identities.
- [x] Logical object completeness across model-parallel ranks.
- [x] Separate physical rank keys from logical completion identities.
- [x] Store/load statistics and transferred-byte counters.
- [x] KV stored and removed events.
- [x] UMBP eviction detection.
- [x] Single-GPU Llama DRAM restore correctness.
- [x] Four-GPU TP=4 Llama DRAM restore correctness.

Layer-wise store remains disabled. MORI ranged put currently requires the ranges
in one put to tile the complete object. Independent per-layer puts do not meet
that contract and must not be advertised as supported.

## DRAM Configuration

### Building MORI with the required API

The embedded adapter needs `mori.cpp.UMBPClient`, `UMBPConfig`, and
`MemoryLocationType`, including the client's `register_memory`,
`deregister_memory`, `batch_get_ranges_into_ptr`, `batch_put_ranges_from_ptr`,
`batch_exists`, `flush`, and `clear` methods. A MORI package without its UMBP
bindings is insufficient, even if its version is recent.

The validated ROCm dependency is MORI v1.2.4, base commit
`f39fa765e897c95995957d0518908cc982a88fc2`, plus the bundled
[`mori-no-master-reset.patch`](../../tools/umbp/mori-no-master-reset.patch).
The repair was authored as MORI commit
`cd9772b5944976926950f54ebbc4794077ffde54` and includes a repeated native
store/clear/store/restore regression. Its source tree is
`99e1f525ff0c2e9c0ba585127c30b37fcec92b4f`; applying the patch with different
committer metadata can change the commit ID while preserving this tree.

The unpatched base leaves embedded allocations blocked with `CLEAR_PENDING`
after `clear()`: cache misses still recompute, but subsequent stores fail.
The repair acknowledges local clear when no master is configured and preserves
the master-backed acknowledgment path. Use the repaired dependency for cache
reset support.

From the vLLM repository root, in an isolated Ubuntu development container with
the virtual environment and ROCm toolchain available:

```bash
apt-get update
apt-get install -y libgrpc-dev libgrpc++-dev libprotobuf-dev \
  protobuf-compiler protobuf-compiler-grpc
git clone --branch v1.2.4 --depth 1 --recursive https://github.com/ROCm/mori.git
test "$(git -C mori rev-parse HEAD)" = f39fa765e897c95995957d0518908cc982a88fc2
git -C mori am ../tools/umbp/mori-no-master-reset.patch
test "$(git -C mori rev-parse 'HEAD^{tree}')" = 99e1f525ff0c2e9c0ba585127c30b37fcec92b4f
BUILD_UMBP=ON GPU_TARGETS=gfx950 PYTORCH_ROCM_ARCH=gfx950 \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps ./mori
.venv/bin/python -c 'from mori.cpp import UMBPClient, UMBPConfig, MemoryLocationType; assert all(hasattr(UMBPClient, m) for m in ("register_memory", "deregister_memory", "batch_get_ranges_into_ptr", "batch_put_ranges_from_ptr", "batch_exists", "flush", "clear"))'
.venv/bin/python -m pytest -q mori/tests/python/umbp/test_umbp_client_ptr.py
```

Use the target architecture of the machine in place of `gfx950`. Building from
source also avoids the newer glibc requirement of published Linux wheels.

### Qwen3.8 Flash Next on ROCm

The AMD Qwen4Exp QSA implementation accepts `UMBPStoreConnector`. Use
`--enable-prefix-caching --mamba-cache-mode align
--no-disable-hybrid-kv-cache-manager`, BF16 main KV, and full-block hits
(`enable_partial_hash_hits=false`). Hybrid loads complete before the request
resumes, even if `load_async=false` was requested. On the validated ROCm
7.2.3 / AITER 0.1.24.post1 image, select Triton MoE as in the recipe below: the default
AITER MoE warmup rejected this model's TP=4 GEMM shape. TP=8 with explicit
AITER MoE also failed during weight preparation: the local expert width of 80
does not satisfy the weight shuffle's multiple-of-32 requirement. The same
failure occurs without UMBP. The model's separate attention backend is
`QWEN4_EXP_QSA_TRITON`.

UMBP stores the main attention cache, compressed QSA keys, GDN state, and PLE
short-convolution state. The raw-key circular buffer is deliberately not
hash-addressed. Full-block hits end on compression boundaries, allowing that
scratch ring to rebuild without reading a previous request's keys. Align-mode
Mamba snapshots come from core's boundary hand-offs rather than live state slots.

Use a distinct `lookup_dir` for each embedded engine and DP replica. The current
embedded socket namespace is not automatically isolated across engines or DP
replicas. PP, context parallelism, speculative decoding, multimodal restore, and
partial-hash QSA restores are outside this recipe's validation scope.

The opt-in real-weight regression accepts a pinned local model and additional
engine arguments:

```bash
ROCR_VISIBLE_DEVICES=0,1,2,3 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=0 \
VLLM_UMBP_HYBRID_TEST_MODEL=/models/Qwen3.8-Flash-Next \
VLLM_UMBP_HYBRID_TEST_ARGS='{"tensor_parallel_size":4,"mamba_cache_mode":"align","disable_hybrid_kv_cache_manager":false,"language_model_only":true,"kernel_config":{"moe_backend":"triton"}}' \
  .venv/bin/python -m pytest -q \
  tests/v1/kv_connector/umbp_integration/test_mori_token_e2e.py \
  -k hybrid_restore
```

It checks retrieval at 1,025, 4,099, and 8,193 prompt tokens with both eager and
lazy offload. Cold, native-prefix, and external-prefix executions must produce
identical tokens and the expected answer. External and repeated native-prefix
log probabilities must agree with the native-prefix baseline within 0.01.
Cacheable filler requests apply GPU eviction pressure before a GPU-cache reset;
completed external-load counters must increase with no transfer failures. A final
connector reset must force recomputation, permit fresh successful stores, and
allow another external restore after clearing the GPU cache. Each post-reset
stage must remain free of transfer failures; restored output is compared with
a matched native-prefix baseline. `VLLM_UMBP_EVIDENCE_DIR` optionally retains the
responses and transfer-counter snapshots.

This focused test does not establish general accuracy or bitwise parity. On the
validated BF16 stack, highly repetitive long prompts showed probability variation
and occasional greedy-token changes even during native GPU prefix reuse. Those
workloads need additional model-backend evaluation. A focused trace found
different ROCm QSA top-k ordering despite identical scores and selected token
sets. Test-only canonical ordering removed the layer-3 QSA and downstream
differences for three repetitive prompts, including UMBP restores. Production
ordering remains unchanged; this counterfactual is neither a performance result
nor evidence of broad model accuracy. The recipe uses `enforce_eager=True`;
eager/lazy offloading describes the transfer policy, not graph execution.

The following embedded options are supported:

```json
{
  "mode": "embedded",
  "total_capacity_bytes": 68719476736,
  "dram_high_watermark": 0.9,
  "dram_low_watermark": 0.7,
  "dram_use_hugepages": true,
  "dram_hugepage_size": 2097152,
  "dram_numa_node": -1,
  "dram_prefault": true,
  "dram_use_shared_memory": false,
  "dram_shm_name": "umbp-dram"
}
```

`capacity_bytes` configures capacity per rank. `total_capacity_bytes` configures
a total budget that is divided by the participating TP/PP/PCP/DCP rank count.
The two options are mutually exclusive.

## Comparison with Simple CPU Offloading

This comparison covers only the CPU backend of `SimpleCPUOffloadConnector`.
Disk offloading is outside the embedded DRAM runtime's scope.

| Capability | UMBP embedded DRAM | Simple CPU offloading |
| --- | --- | --- |
| Host-memory owner | MORI/UMBP | vLLM worker |
| Data organization | Hash-keyed KV objects | Fixed CPU block slots |
| GPU-to-CPU transfer | MORI ranged pointer API | vLLM DMA copy backend |
| CPU allocation | UMBP DRAM allocator | PyTorch CPU tensors with pinning |
| Capacity control | Per-rank bytes and watermarks | Fixed slot capacity |
| NUMA placement | Configurable | No equivalent connector option |
| Huge pages | Configurable | No equivalent connector option |
| Prefaulting | Configurable | No equivalent connector option |
| Rank completeness | Explicit TP/PP/PCP/DCP objects | Internal block ownership |
| Partial ranged objects | Supported | Primarily full-block copies |
| Layer-wise load | Supported | Whole-load asynchronous submission |
| Layer-wise store | Disabled | Not a true per-layer store path |
| Eager offload | Supported and validated | Supported |
| Lazy offload | Free-queue based and tested | Free-queue based and tested |
| Preemption | Implemented, needs stress E2E | More mature coverage |
| HMA integration | Supported | Supported |
| MultiConnector coverage | Needs system validation | NIXL combinations tested |
| Model coverage | Llama, single GPU and TP=4 | Broader model/eval coverage |
| Performance baselines | Not complete | Existing latency/performance tests |

## Remaining Development Work

- [x] Document `capacity_bytes` explicitly as per-rank capacity in the public
  connector configuration documentation and startup logs.
- [x] Add `total_capacity_bytes` and divide it across participating workers.
- [x] Add authoritative per-rank lookup diagnostics for logical misses.
- [x] Add structured diagnostics for unavailable rank sockets and missing keys.
- [ ] Add configurable transfer concurrency and queue-depth tuning if MORI
  measurements show the current worker pool is limiting throughput.
- [ ] Tune `lazy_offload_max_blocks` against GPU pressure and transfer latency.
- [x] Verify that DRAM watermark eviction produces timely and complete KV
  removal events under sustained load.
- [ ] Harden cancellation when GPU blocks are reused immediately after
  preemption.
- [ ] Validate memory cleanup and socket cleanup across repeated engine startup
  and shutdown.
- [ ] Keep layer-wise store disabled until MORI supports atomic staged partial
  object assembly, or another full-object-safe protocol is implemented.

## Remaining Correctness Tests

- [x] Single-GPU Llama store, eviction, DRAM restore, and token equality.
- [x] TP=4 Llama store, eviction, logical completeness, DRAM restore, and token
  equality.
- [x] Four-GPU raw MORI range store/load round trip.
- [x] Rank-local physical keys with canonical logical completion aggregation.
- [x] Lazy offload with a real model on one GPU.
- [x] Lazy offload with TP=4.
- [ ] Store cancellation followed by immediate GPU block reuse.
- [ ] Partial failure where one TP rank fails to publish an object.
- [x] Partial eviction where one TP rank loses an object before lookup.
- [x] Repeated eviction and restore cycles under a deliberately small DRAM
  capacity.
- [ ] PP=2 serving-level restore (unsupported in embedded mode).
- [ ] TP=2 plus PP=2 serving-level restore (unsupported in embedded mode).
- [ ] DCP serving-level restore.
- [x] Hybrid attention and Mamba model restore (Qwen3.8 Flash Next, TP=4).
- [x] Multiple prefix-cacheable KV groups (Qwen3.8 Flash Next).
- [ ] Long-running concurrent request stress test.
- [ ] Engine restart and cleanup test for lookup sockets and registered memory.
- [ ] MultiConnector integration where UMBP embedded is combined with another
  connector.

## Remaining Performance Tests

- [ ] Measure GPU-to-DRAM bandwidth by transfer size and batch size.
- [ ] Measure DRAM-to-GPU bandwidth by transfer size and batch size.
- [ ] Compare restore latency with `SimpleCPUOffloadConnector`.
- [ ] Compare time-to-first-token after an external-cache hit.
- [ ] Measure transfer overlap with model compute.
- [ ] Sweep embedded `num_workers`.
- [ ] Compare local and remote NUMA placement.
- [ ] Compare huge pages enabled and disabled.
- [ ] Compare prefaulting enabled and disabled.
- [ ] Measure eviction overhead at different high and low watermarks.
- [ ] Measure p50, p95, and p99 latency under concurrent load.

## Development and Test Image

The current local development image with `total_capacity_bytes` support is:

```text
rocm/pytorch-private:vllm_umbp_embedded_lazy_gfx950_dev_20260918
```

This image also includes Ruff 0.14.0 for source and test lint checks.

The reusable MI355X development image contains the current UMBP embedded
source, MORI bindings, and vLLM ROCm extensions compiled for `gfx950`:

```text
rocm/pytorch-private:vllm_umbp_embedded_dramcfg_gfx950_20260918
```

Registry digest:

```text
sha256:b5bf4e18b2e64588178999969257db55a6de376c8fee77fcb2eeee05c1633ed6
```

The image has been used on Mi355 GPU 0-3. The TP=4 Llama validation restored
192 cached prompt tokens and produced the same output tokens as the cold run.

Example launch options:

```bash
docker run --rm --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /data/models:/data/models:ro \
  rocm/pytorch-private:vllm_umbp_embedded_dramcfg_gfx950_20260918 \
  <test-command>
```

## Current Assessment

Within the intentionally narrow scope of local UMBP-managed DRAM offloading:

- Core store/load correctness: approximately 90% complete.
- Basic parity with the Simple CPU backend: approximately 85% complete.
- Production validation and performance maturity: approximately 65-70%
  complete.

The primary remaining work is validation and hardening around lazy offload,
eviction, preemption, additional parallel layouts, hybrid KV caches, and
performance under sustained concurrent load.
