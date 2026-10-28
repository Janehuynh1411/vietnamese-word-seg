"""Direct Preference Optimization for PhoBERT word segmentation."""

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from llm.phobert import PhoBERTWordSegmenter, align_labels_with_tokens, get_device


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bradley-Terry DPO loss (Rafailov et al., 2023):

        L = -E[ log σ( β * ((log π(y_w|x) - log π_ref(y_w|x))
                           - (log π(y_l|x) - log π_ref(y_l|x))) ) ]

    Returns (loss, reward_margin) where reward_margin is the mean implied
    reward gap between chosen and rejected sequences.
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    reward_margin = (beta * (chosen_logratios - rejected_logratios)).detach().mean()
    return loss, reward_margin


def sequence_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Average per-token log-probability of B/I labels under token-classification logits.
    Positions with label == -100 are ignored. Returns a tensor of shape [batch].
    """
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels = safe_labels.masked_fill(labels == -100, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    token_log_probs = token_log_probs * mask
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    return token_log_probs.sum(dim=-1) / lengths


def assemble_dpo_objective(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    sft_loss: torch.Tensor,
    beta: float = 0.1,
    sft_anchor_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One DPO step: preference loss plus a cross-entropy anchor on the chosen tags."""
    policy_chosen = sequence_log_probs_from_logits(policy_logits, chosen_labels)
    policy_rejected = sequence_log_probs_from_logits(policy_logits, rejected_labels)
    ref_chosen = sequence_log_probs_from_logits(ref_logits, chosen_labels)
    ref_rejected = sequence_log_probs_from_logits(ref_logits, rejected_labels)
    dpo_loss, reward_margin = compute_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
    )
    total = dpo_loss + sft_anchor_weight * sft_loss
    return total, dpo_loss, reward_margin


class PhoBERTDPODataset(Dataset):
    """Preference pairs aligned to PhoBERT BPE subwords."""

    def __init__(
        self,
        preference_pairs: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        self.samples = []
        for pair in preference_pairs:
            syllables = pair.get("syllables") or []
            chosen_tags = pair.get("chosen_tags") or []
            rejected_tags = pair.get("rejected_tags") or []
            if not syllables or len(chosen_tags) != len(syllables) or len(rejected_tags) != len(syllables):
                continue
            chosen = align_labels_with_tokens(
                tokenizer, syllables, labels=chosen_tags, label2id=label2id, max_length=max_length
            )
            rejected = align_labels_with_tokens(
                tokenizer, syllables, labels=rejected_tags, label2id=label2id, max_length=max_length
            )
            self.samples.append(
                {
                    "input_ids": chosen["input_ids"],
                    "attention_mask": chosen["attention_mask"],
                    "chosen_labels": chosen["labels"],
                    "rejected_labels": rejected["labels"],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def collate_fn(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)

        def pad(values, fill):
            return [row + [fill] * (width - len(row)) for row in values]

        return {
            "input_ids": torch.tensor(pad([item["input_ids"] for item in batch], self.pad_id), dtype=torch.long),
            "attention_mask": torch.tensor(
                pad([item["attention_mask"] for item in batch], 0), dtype=torch.long
            ),
            "chosen_labels": torch.tensor(
                pad([item["chosen_labels"] for item in batch], -100), dtype=torch.long
            ),
            "rejected_labels": torch.tensor(
                pad([item["rejected_labels"] for item in batch], -100), dtype=torch.long
            ),
        }


class PhoBERTDPOTrainer:
    """
    DPO trainer for PhoBERT word segmentation.

    The policy is initialized from a supervised checkpoint. A frozen copy is the
    reference policy π_ref. The objective is the Bradley-Terry DPO loss plus a
    small supervised anchor on the chosen B/I tags so the tagger does not drift
    off the token-classification format.
    """

    def __init__(
        self,
        base_checkpoint: str = "vinai/phobert-base",
        beta: float = 0.1,
        sft_anchor_weight: float = 0.1,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.device = get_device(device)
        self.beta = beta
        self.sft_anchor_weight = sft_anchor_weight
        self.max_length = max_length
        self.policy_segmenter = PhoBERTWordSegmenter(
            model_name_or_path=base_checkpoint,
            max_length=max_length,
            device=str(self.device),
        )
        self.ref_model = copy.deepcopy(self.policy_segmenter.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train(
        self,
        train_pairs: List[dict],
        val_pairs: Optional[List[dict]] = None,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        save_dir: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        dataset = PhoBERTDPODataset(
            train_pairs,
            tokenizer=self.policy_segmenter.tokenizer,
            label2id=self.policy_segmenter.label2id,
            max_length=self.max_length,
        )
        if len(dataset) == 0:
            raise ValueError("No usable preference pairs for PhoBERT-DPO training.")

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        optimizer = AdamW(self.policy_segmenter.model.parameters(), lr=learning_rate)
        history: Dict[str, List[float]] = {"dpo_loss": [], "reward_margin": [], "total_loss": []}
        steps_done = 0
        stop = False

        for epoch in range(1, epochs + 1):
            self.policy_segmenter.model.train()
            total_loss = 0.0
            total_dpo = 0.0
            total_margin = 0.0
            steps = 0
            for batch in loader:
                if max_steps is not None and steps_done >= max_steps:
                    stop = True
                    break
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                supervised = self.policy_segmenter.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=chosen_labels,
                )
                policy_logits = supervised.logits
                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

                loss, dpo_loss, margin = assemble_dpo_objective(
                    policy_logits,
                    ref_logits,
                    chosen_labels,
                    rejected_labels,
                    supervised.loss,
                    beta=self.beta,
                    sft_anchor_weight=self.sft_anchor_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_segmenter.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_dpo += float(dpo_loss.item())
                total_margin += float(margin.item())
                steps += 1
                steps_done += 1

            if steps:
                history["total_loss"].append(total_loss / steps)
                history["dpo_loss"].append(total_dpo / steps)
                history["reward_margin"].append(total_margin / steps)
                print(
                    f"Epoch {epoch}/{epochs} | total_loss={history['total_loss'][-1]:.4f} "
                    f"| dpo_loss={history['dpo_loss'][-1]:.4f} "
                    f"| reward_margin={history['reward_margin'][-1]:.4f}"
                )
            if stop:
                break

        if val_pairs:
            accuracy = self.evaluate_preference_accuracy(val_pairs, batch_size=batch_size)
            history["val_pref_acc"] = [accuracy]
            print(f"Validation preference accuracy: {accuracy:.4f}")

        if save_dir:
            self.policy_segmenter.save(save_dir)
        return history

    def evaluate_preference_accuracy(self, pairs: List[dict], batch_size: int = 8) -> float:
        dataset = PhoBERTDPODataset(
            pairs,
            tokenizer=self.policy_segmenter.tokenizer,
            label2id=self.policy_segmenter.label2id,
            max_length=self.max_length,
        )
        if len(dataset) == 0:
            return 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.policy_segmenter.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)
                policy_logits = self.policy_segmenter.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
                policy_chosen = sequence_log_probs_from_logits(policy_logits, chosen_labels)
                policy_rejected = sequence_log_probs_from_logits(policy_logits, rejected_labels)
                ref_chosen = sequence_log_probs_from_logits(ref_logits, chosen_labels)
                ref_rejected = sequence_log_probs_from_logits(ref_logits, rejected_labels)
                gaps = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
                correct += int((gaps > 0).sum().item())
                total += int(gaps.shape[0])
        return correct / total if total else 0.0


def _compound_tokens(segmented: str) -> set:
    return {token.lower() for token in segmented.split() if "_" in token}


def build_llm_preference_memory(train_pairs: List[dict], save_path: str) -> Dict[str, object]:
    """
    Extract compound joins the preference data wants kept, and compounds it rejects,
    for injection into the Tier-3 LLM prompt.
    """
    preferred = set()
    rejected = set()
    for pair in train_pairs:
        chosen = pair.get("chosen") or ""
        rejected_text = pair.get("rejected") or ""
        chosen_compounds = _compound_tokens(chosen)
        rejected_compounds = _compound_tokens(rejected_text)
        preferred.update(chosen_compounds - rejected_compounds)
        rejected.update(rejected_compounds - chosen_compounds)

    payload = {
        "preferred_compounds": sorted(preferred),
        "rejected_compounds": sorted(rejected),
        "preferred_count": len(preferred),
        "rejected_count": len(rejected),
    }
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_preference_pairs(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    pairs: List[dict] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload.get("pairs") or payload.get("preferences") or [])
    return list(payload)
