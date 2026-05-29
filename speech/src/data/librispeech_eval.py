"""
=============================================================================
LibriSpeech Utilities
=============================================================================
"""

from datasets import load_dataset


def load_librispeech_subset(
    max_samples=20,
    cache_dir=None,
):

    print(
        "[LibriSpeech] Loading transcript subset..."
    )

    ds = load_dataset(

        "openslr/librispeech_asr",

        "clean",

        split="test.clean",

        streaming=True,

        cache_dir=cache_dir,
    )

    ds = ds.take(max_samples)

    return ds