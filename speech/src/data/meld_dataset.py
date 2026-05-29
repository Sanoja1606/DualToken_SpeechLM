"""
=============================================================================
MELD emotion labels → our 7-class mapping:
    neutral    → 0
    joy        → 1 (happy)
    sadness    → 2
    anger      → 3
    fear       → 4
    surprise   → 5
    disgust    → 6
=============================================================================
"""

from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import Dataset

# ── Label mappings ────────────────────────────────────────────────────────────

MELD_EMOTION_MAP: Dict[str, int] = {
    "neutral" : 0,
    "joy"     : 1,
    "sadness" : 2,
    "anger"   : 3,
    "fear"    : 4,
    "surprise": 5,
    "disgust" : 6,
}

# MELD dialogue_act heuristics based on sentiment + emotion
MELD_SENTIMENT_TO_ACT: Dict[str, str] = {
    "positive": "commissive",
    "neutral" : "inform",
    "negative": "directive",
}


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MELDDataset(Dataset):
    """
    Wraps a MELD split mapped from tabular records.

    Each item returns:
        {
            "utterance"   : str,
            "speaker"     : str,
            "emotion_id"  : int,   # 0-6 mapped from MELD emotion string
            "sentiment"   : str,   # "positive" / "neutral" / "negative"
            "dialogue_act": str,   # heuristic act label
            "season"      : int,
            "episode"     : int,
            "dialogue_id" : int,
        }
    """

    def __init__(self, records_list: List[Dict]):
        """
        Args:
            records_list : A parsed list of dictionary rows from the dataset
        """
        self.records: List[Dict] = []
        for row in records_list:
            # Safely get and parse emotion features
            emo_str = str(row.get("Emotion", row.get("emotion", "neutral"))).lower().strip()
            emotion_id = MELD_EMOTION_MAP.get(emo_str, 0)
            
            # Safely get and parse sentiment features
            sentiment = str(row.get("Sentiment", row.get("sentiment", "neutral"))).lower().strip()
            act_label = MELD_SENTIMENT_TO_ACT.get(sentiment, "inform")

            self.records.append({
                "utterance"   : str(row.get("Utterance", row.get("utterance", ""))),
                "speaker"     : str(row.get("Speaker", row.get("speaker", ""))),
                "emotion_id"  : emotion_id,
                "sentiment"   : sentiment,
                "dialogue_act": act_label,
                "season"      : int(row.get("Season", row.get("season", 0))),
                "episode"     : int(row.get("Episode", row.get("episode", 0))),
                "dialogue_id" : int(row.get("Dialogue_ID", row.get("dialogue_id", 0))),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_meld_splits(
    hf_dataset_name: str = "declare-lab/MELD",
    trust_remote_code: bool = False,
) -> Tuple[MELDDataset, MELDDataset, MELDDataset]:
    """
    Bypasses the heavy 11GB multimodal download by reading text-only CSV data
    directly from the MELD repository. Safe against out-of-disk crashes.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas package not found. Install with: pip install pandas")

    print("[MELD] Streaming text-only configs safely from repository source...")
    
    train_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
    val_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv"
    test_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"

    # Read records out of stream directly into transient virtual memory
    df_train = pd.read_csv(train_url, encoding="utf-8")
    df_val = pd.read_csv(val_url, encoding="utf-8")
    df_test = pd.read_csv(test_url, encoding="utf-8")

    # Construct wrapped datasets
    train_ds = MELDDataset(df_train.to_dict(orient="records"))
    val_ds   = MELDDataset(df_val.to_dict(orient="records"))
    test_ds  = MELDDataset(df_test.to_dict(orient="records"))

    print(f"[MELD] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ── Collator ──────────────────────────────────────────────────────────────────

def meld_collate(batch: List[Dict]) -> Dict:
    """
    Minimal collator — returns utterances + emotion_ids for SER training.
    """
    return {
        "texts"        : [item["utterance"] for item in batch],
        "emotion_ids"  : [item["emotion_id"] for item in batch],
        "dialogue_acts": [item["dialogue_act"] for item in batch],
    }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_meld_splits()
    print("Sample:", train_ds[0])