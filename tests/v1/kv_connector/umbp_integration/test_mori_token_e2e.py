# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-level correctness tests for the MORI-backed UMBP runtime."""

import json
import os
import time
from pathlib import Path

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

pytest.importorskip("mori.cpp")

if not current_platform.is_cuda_alike():
    pytest.skip("requires CUDA or ROCm", allow_module_level=True)


def _flush_gpu_cache(llm: LLM, sampling_params: SamplingParams) -> None:
    cache_config = llm.llm_engine.vllm_config.cache_config
    total_tokens = int(cache_config.num_gpu_blocks * cache_config.block_size * 1.5)
    # Sub-block prompts never become prefix-cache entries and may repeatedly
    # reuse the same free pages without pressuring lazy offload.
    tokens_per_request = max(512, cache_config.block_size + 1)
    num_requests = (total_tokens + tokens_per_request - 1) // tokens_per_request
    for start in range(0, num_requests, 16):
        prompts = [
            TokensPrompt(
                prompt_token_ids=[index // 100 + 3, index % 100 + 3]
                + [(index % 100) + 3] * (tokens_per_request - 2)
            )
            for index in range(start, min(start + 16, num_requests))
        ]
        llm.generate(prompts, sampling_params, use_tqdm=False)


@pytest.mark.skipif(
    "VLLM_UMBP_TEST_MODEL" not in os.environ,
    reason="set VLLM_UMBP_TEST_MODEL to a local model path",
)
def test_mori_external_kv_preserves_generated_tokens(tmp_path):
    model = os.environ["VLLM_UMBP_TEST_MODEL"]
    llm = LLM(
        model=model,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
        max_model_len=1024,
        kv_cache_memory_bytes=16 << 20,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "capacity_bytes": 256 << 20,
                "lookup_dir": str(tmp_path),
                "num_workers": 2,
                "timeout_ms": 60000,
                "load_async": False,
            },
        ),
    )
    sampling_params = SamplingParams(max_tokens=4, temperature=0)
    prompt = "The first ten natural numbers are " * 32

    cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    _flush_gpu_cache(llm, SamplingParams(max_tokens=1, temperature=0))
    time.sleep(1)
    restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

    assert restored.outputs[0].token_ids == cold.outputs[0].token_ids
    assert restored.num_cached_tokens > 0


@pytest.mark.skipif(
    "VLLM_UMBP_HYBRID_TEST_MODEL" not in os.environ,
    reason="set VLLM_UMBP_HYBRID_TEST_MODEL to a local hybrid model",
)
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_mori_hybrid_restore_after_gpu_reset(tmp_path, lazy):
    """Real weights, fresh GPU cache, multi-token output and external-load proof."""
    options = json.loads(os.environ.get("VLLM_UMBP_HYBRID_TEST_ARGS", "{}"))
    llm = LLM(
        model=os.environ["VLLM_UMBP_HYBRID_TEST_MODEL"],
        enforce_eager=True,
        max_model_len=9216,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        kv_cache_memory_bytes=512 << 20,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "capacity_bytes": 4 << 30,
                "lookup_dir": str(tmp_path),
                "timeout_ms": 60000,
                "load_async": False,
                "lazy_offload": lazy,
            },
        ),
        **options,
    )

    def reset_cache(reset_connector):
        # Completed generation may still have asynchronous stores pinning pages.
        deadline = time.monotonic() + 30
        while not llm.reset_prefix_cache(reset_connector=reset_connector):
            assert time.monotonic() < deadline, "UMBP stores did not drain"
            time.sleep(0.1)

    def metrics():
        return [
            {"name": m.name, "labels": m.labels, "value": m.value}
            for m in llm.get_metrics()
            if "umbp" in m.name or "external_prefix_cache" in m.name
        ]

    def count(snapshot, name, operation=None):
        return sum(
            m["value"]
            for m in snapshot
            if name in m["name"]
            and (operation is None or m["labels"].get("operation") == operation)
        )

    def result(output):
        sample = output.outputs[0]
        return {
            "prompt_tokens": len(output.prompt_token_ids),
            "token_ids": sample.token_ids,
            "text": sample.text,
            "finish_reason": sample.finish_reason,
            "cached_tokens": output.num_cached_tokens,
            "logprobs": [
                lp[token].logprob
                for lp, token in zip(sample.logprobs, sample.token_ids)
            ],
        }

    report = {"lazy": lazy, "cases": []}
    try:
        tokenizer = llm.get_tokenizer()
        words = [
            "apple",
            "cedar",
            "ocean",
            "violet",
            "copper",
            "willow",
            "stone",
            "amber",
        ]
        records = "\n".join(
            f"Record {i:04d}: {words[i % 8]} shipment, quantity {(i * 37) % 997}, "
            f"warehouse {words[(i // 8) % 8]}, reference {(i * 7919) % 100003}."
            for i in range(1000)
        )
        filler = tokenizer.encode(records, add_special_tokens=False)
        sampling = SamplingParams(temperature=0, max_tokens=16, logprobs=1)
        for index, length in enumerate((1025, 4099, 8193)):
            header = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Catalog {index}. The verification phrase is "
                            f"silver maple 7042.\n{records}\n"
                            "Report the verification phrase only."
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix, suffix = header.split(records)
            first = tokenizer.encode(prefix, add_special_tokens=False)
            last = tokenizer.encode(suffix, add_special_tokens=False)
            ids = first + filler[: length - len(first) - len(last)] + last
            prompt = TokensPrompt(prompt_token_ids=ids)
            cold = llm.generate(prompt, sampling, use_tqdm=False)[0]
            warm_before = metrics()
            warm = llm.generate(prompt, sampling, use_tqdm=False)[0]
            warm_repeat = llm.generate(prompt, sampling, use_tqdm=False)[0]
            warm_after = metrics()
            # Lazy stores are driven by pressure on the free queue; these also
            # overwrite freed GPU pages before the replay.
            _flush_gpu_cache(llm, SamplingParams(max_tokens=1, temperature=0))
            reset_cache(reset_connector=False)
            before = metrics()
            restored = llm.generate(prompt, sampling, use_tqdm=False)[0]
            after = metrics()
            a, b, local = result(cold), result(restored), result(warm)
            case = {
                "cold": a,
                "local_prefix": local,
                "local_repeat": result(warm_repeat),
                "warm_before": warm_before,
                "warm_after": warm_after,
                "restored": b,
                "before": before,
                "after": after,
            }
            report["cases"].append(case)
            print(
                "Completed hybrid case",
                length,
                "cached tokens",
                b["cached_tokens"],
                flush=True,
            )
        for case in report["cases"]:
            a, b, local = case["cold"], case["restored"], case["local_prefix"]
            before, after = case["before"], case["after"]
            assert b["token_ids"] == a["token_ids"]
            assert b["finish_reason"] == a["finish_reason"]
            assert b["cached_tokens"] > 0
            assert count(after, "umbp_transfer_completed", "load") > count(
                before, "umbp_transfer_completed", "load"
            )
            assert count(after, "umbp_transfer_failed") == 0
            # Native prefix reuse is the matching execution-path baseline:
            # recurrent prefill kernels may differ numerically from cold chunks.
            assert local["token_ids"] == a["token_ids"]
            assert b["cached_tokens"] == local["cached_tokens"]
            assert b["text"].strip() == "silver maple 7042"
            assert count(
                case["warm_after"], "umbp_transfer_completed", "load"
            ) == count(case["warm_before"], "umbp_transfer_completed", "load")
            assert case["local_repeat"]["token_ids"] == local["token_ids"]
            # BF16 sparse attention is not bitwise invariant under prefix reuse.
            # Bound the native control and external replay with the same tolerance.
            assert case["local_repeat"]["logprobs"] == pytest.approx(
                local["logprobs"], abs=0.01
            )
            assert b["logprobs"] == pytest.approx(local["logprobs"], abs=0.01)
        reset_cache(reset_connector=True)
        before_clear_replay = metrics()
        cleared = llm.generate(prompt, sampling, use_tqdm=False)[0]
        report["after_connector_reset"] = result(cleared)
        assert cleared.num_cached_tokens == 0
        assert cleared.outputs[0].token_ids == report["cases"][-1]["cold"]["token_ids"]
        assert count(metrics(), "umbp_transfer_completed", "load") == count(
            before_clear_replay, "umbp_transfer_completed", "load"
        )
        # Regression: connector clear must permit fresh stores and restores.
        # Match the native cached execution path before testing external replay.
        before_reset_native = metrics()
        native_after_reset = llm.generate(prompt, sampling, use_tqdm=False)[0]
        after_reset_native = metrics()
        if lazy:
            _flush_gpu_cache(llm, SamplingParams(max_tokens=1, temperature=0))
        # Waiting for GPU-only reset drains stores without clearing the backend.
        reset_cache(reset_connector=False)
        after_reset_store = metrics()
        report["reset_cycle"] = {
            "before_store": before_clear_replay,
            "after_store": after_reset_store,
            "cold_after_clear": result(cleared),
            "native_after_clear": result(native_after_reset),
            "before_native": before_reset_native,
            "after_native": after_reset_native,
        }
        replay = llm.generate(prompt, sampling, use_tqdm=False)[0]
        after_reset_restore = metrics()
        report["reset_cycle"].update(
            {
                "restored": result(replay),
                "after_restore": after_reset_restore,
            }
        )
        assert count(after_reset_native, "umbp_transfer_failed") == 0
        assert count(after_reset_store, "umbp_transfer_failed") == 0
        assert count(after_reset_store, "umbp_transfer_completed", "store") > count(
            before_clear_replay, "umbp_transfer_completed", "store"
        ), "no successful UMBP stores after connector reset"
        assert count(after_reset_restore, "umbp_transfer_failed") == 0
        assert replay.num_cached_tokens == native_after_reset.num_cached_tokens > 0
        assert count(after_reset_restore, "umbp_transfer_completed", "load") > count(
            after_reset_store, "umbp_transfer_completed", "load"
        ), "no UMBP restore after post-reset stores"
        assert count(after_reset_native, "umbp_transfer_completed", "load") == count(
            before_reset_native, "umbp_transfer_completed", "load"
        )
        assert replay.outputs[0].token_ids == native_after_reset.outputs[0].token_ids
        assert replay.outputs[0].token_ids == cleared.outputs[0].token_ids
        assert result(replay)["logprobs"] == pytest.approx(
            result(native_after_reset)["logprobs"], abs=0.01
        )
    finally:
        destination = Path(os.environ.get("VLLM_UMBP_EVIDENCE_DIR", str(tmp_path)))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"hybrid-{'lazy' if lazy else 'eager'}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        llm.llm_engine.engine_core.shutdown()
