__all__ = [
    "DFlashDraftModel",
    "PaperSuffixNode",
    "PaperSuffixPrediction",
    "SuffixMatcher",
    "SuffixPrediction",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark_common import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    if name in {"PaperSuffixNode", "PaperSuffixPrediction", "SuffixMatcher", "SuffixPrediction"}:
        from .suffix_decoding import PaperSuffixNode, PaperSuffixPrediction, SuffixMatcher, SuffixPrediction

        return {
            "PaperSuffixNode": PaperSuffixNode,
            "PaperSuffixPrediction": PaperSuffixPrediction,
            "SuffixMatcher": SuffixMatcher,
            "SuffixPrediction": SuffixPrediction,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
