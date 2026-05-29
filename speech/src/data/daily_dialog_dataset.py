"""
=============================================================================
DailyDialog Dataset Loader 
=============================================================================
"""

import os
import ast
import torch
import pandas as pd
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

NUM_DIALOGUE_ACTS = 6

DA_QUESTION    = 0
DA_UNCERTAINTY = 1
# Mapped tags: 0=question, 1=uncertainty, 2=negation, 3=affirmation
DA_NEGATION    = 2
DA_AFFIRMATION = 3
DA_TOPIC_SHIFT = 4
DA_EMOTION     = 5

def _dd_act_to_multihot(act_int: int) -> torch.Tensor:
    vec = torch.zeros(NUM_DIALOGUE_ACTS, dtype=torch.float32)
    if act_int == 2:   # question
        vec[DA_QUESTION] = 1.0
    elif act_int == 1: # inform
        vec[DA_AFFIRMATION] = 1.0
    elif act_int == 3: # directive
        vec[DA_QUESTION] = 0.5          
        vec[DA_AFFIRMATION] = 0.5
    elif act_int == 4: # commissive
        vec[DA_AFFIRMATION] = 1.0
    return vec

# ── Dataset ───────────────────────────────────────────────────────────────────

class DailyDialogDataset(Dataset):
    _EMO_MAP = {0: 0, 1: 3, 2: 6, 3: 4, 4: 1, 5: 2, 6: 5}

    def __init__(self, df: pd.DataFrame):
        self.records: List[Dict] = []
        
        for _, row in df.iterrows():
            # Extract raw cell data blocks
            raw_dialog = row.get('dialog', '')
            raw_act    = row.get('act', '')
            raw_emotion = row.get('emotion', '')

            # Convert literal string lists into executable python arrays safely
            try:
                # Handle space-separated integer arrays like '[3 2 3 4]' safely
                if isinstance(raw_act, str) and ',' not in raw_act:
                    raw_act = raw_act.replace(' ', ', ')
                if isinstance(raw_emotion, str) and ',' not in raw_emotion:
                    raw_emotion = raw_emotion.replace(' ', ', ')

                utterances = ast.literal_eval(str(raw_dialog)) if isinstance(raw_dialog, str) else raw_dialog
                acts       = ast.literal_eval(str(raw_act)) if isinstance(raw_act, str) else raw_act
                emotions   = ast.literal_eval(str(raw_emotion)) if isinstance(raw_emotion, str) else raw_emotion
            except Exception:
                # Skip rows that fail fundamental bracket evaluation parsing
                continue

            # Ensure data components match loop requirements
            if not isinstance(utterances, list) or not isinstance(acts, list):
                continue

            # Unpack every single conversational turn inside the chat array row
            for u, a, e in zip(utterances, acts, emotions):
                utt_str = str(u).strip()
                
                try:
                    act_int = int(a)
                    emo_int = int(e)
                except (ValueError, TypeError):
                    continue

                # ── FIXED CONDITIONAL ──────────────────────────────────────────
                # Filter out completely empty utterances or padding baseline act IDs
                if not utt_str or act_int == 0:
                    continue
                # ───────────────────────────────────────────────────────────────

                self.records.append({
                    "utterance"       : utt_str,
                    "dialogue_act_id" : act_int,
                    "dialogue_act_vec": _dd_act_to_multihot(act_int),
                    "emotion_id"      : self._EMO_MAP.get(emo_int, 0),
                })

    def __len__(self) -> int: 
        return len(self.records)
        
    def __getitem__(self, idx: int) -> Dict: 
        return self.records[idx]

# ── Loader ────────────────────────────────────────────────────────────────────

def load_daily_dialog() -> Tuple[DailyDialogDataset, DailyDialogDataset, DailyDialogDataset]:
    print("[DailyDialog] Locating local CSV files...")
    
    base_path = "/content/speech/src/data/DailyDialog"
    fallback_path = "/content/nlp_project_colabedit/speech/src/data/DailyDialog"
    
    if not os.path.exists(base_path):
        base_path = fallback_path

    train_path = os.path.join(base_path, "train.csv")
    val_path   = os.path.join(base_path, "validation.csv")
    test_path  = os.path.join(base_path, "test.csv")

    print(f"[DailyDialog] Reading arrays from nested tables at: {base_path}")
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)

    train_ds = DailyDialogDataset(df_train)
    val_ds   = DailyDialogDataset(df_val)
    test_ds  = DailyDialogDataset(df_test)

    print(f"[DailyDialog] Local Load Success - Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds

# ── Collator ──────────────────────────────────────────────────────────────────

def daily_dialog_collate(batch: List[Dict]) -> Dict:
    return {
        "texts"              : [item["utterance"] for item in batch],
        "dialogue_act_labels": torch.stack([item["dialogue_act_vec"] for item in batch]),
        "emotion_ids"        : [item["emotion_id"] for item in batch],
    }