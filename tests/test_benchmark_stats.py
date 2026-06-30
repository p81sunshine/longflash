from dflash.benchmark import _summary_from_records


def test_summary_includes_suffix_match_stats():
    summary = _summary_from_records(
        [
            {
                "status": "ok",
                "prompt_tokens": 10,
                "baseline_tpot": 2.0,
                "dflash_tpot": 1.0,
                "speedup": 2.0,
                "mean_acceptance_length": 4.0,
                "acceptance_rate": 0.25,
                "draft_forward_passes": 10,
                "draft_dynamic_yarn_factor": 2.0,
                "suffix_paper_scores": [1.0, 3.0],
                "suffix_paper_tree_sizes": [2, 4],
                "ctx_suffix_match_count": 3,
                "ctx_suffix_match_kept_tokens": 8,
                "ctx_middle_tokens_before_budget": 8,
                "ctx_middle_tokens_after_budget": 4,
                "ctx_middle_budget_dropped_tokens": 4,
                "ctx_hidden_tokens_after": 11,
            },
            {
                "status": "ok",
                "prompt_tokens": 20,
                "baseline_tpot": 4.0,
                "dflash_tpot": 2.0,
                "speedup": 2.0,
                "mean_acceptance_length": 6.0,
                "acceptance_rate": 0.5,
                "draft_forward_passes": 14,
                "draft_dynamic_yarn_factor": 4.0,
                "suffix_paper_scores": [2.0],
                "suffix_paper_tree_sizes": [3],
                "ctx_suffix_match_count": 1,
                "ctx_suffix_match_kept_tokens": 2,
                "ctx_middle_tokens_before_budget": 4,
                "ctx_middle_tokens_after_budget": 4,
                "ctx_middle_budget_dropped_tokens": 0,
                "ctx_hidden_tokens_after": 12,
            },
        ]
    )

    assert summary["mean_ctx_suffix_match_count"] == 2.0
    assert summary["mean_draft_forward_passes"] == 12.0
    assert summary["mean_draft_dynamic_yarn_factor"] == 3.0
    assert summary["min_draft_dynamic_yarn_factor"] == 2.0
    assert summary["max_draft_dynamic_yarn_factor"] == 4.0
    assert summary["mean_suffix_paper_score"] == 2.0
    assert summary["mean_suffix_paper_tree_size"] == 3.0
    assert summary["max_ctx_suffix_match_count"] == 3
    assert summary["total_ctx_suffix_match_count"] == 4
    assert summary["mean_ctx_suffix_match_kept_tokens"] == 5.0
    assert summary["mean_ctx_middle_tokens_before_budget"] == 6.0
    assert summary["mean_ctx_middle_tokens_after_budget"] == 4.0
    assert summary["mean_ctx_middle_budget_dropped_tokens"] == 2.0
    assert summary["mean_ctx_hidden_tokens_after"] == 11.5
