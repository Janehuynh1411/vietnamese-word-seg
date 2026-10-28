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
    beta: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Direct Preference Optimization (DPO) loss (Rafailov et al., 2023):
      L_DPO = -E[ log sigmoid( beta * ((log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))) ) ]
    Returns:
      (loss, reward_margins, preference_accuracies)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margins = chosen_rewards - rejected_rewards
    pref_acc = (reward_margins > 0).float()

    return losses.mean(), reward_margins.mean(), pref_acc.mean()


class PhoBERTDPODataset(Dataset):
    """Dataset of (syllables, chosen_tags, rejected_tags) aligned to PhoBERT BPE subwords."""

    def __init__(
        self,
        preference_pairs: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.samples = []
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        for item in preference_pairs:
            syllables = item.get("syllables", [])
            chosen_tags = item.get("chosen_tags", [])
            rejected_tags = item.get("rejected_tags", [])
            if not syllables or len(syllables) != len(chosen_tags) or len(syllables) != len(rejected_tags):
                continue
            if chosen_tags == rejected_tags:
                continue

            aligned_chosen = align_labels_with_tokens(
                tokenizer=tokenizer,
                tokens=syllables,
                labels=chosen_tags,
                label2id=label2id,
                max_length=max_length,
            )
            aligned_rejected = align_labels_with_tokens(
                tokenizer=tokenizer,
                tokens=syllables,
                labels=rejected_tags,
                label2id=label2id,
                max_length=max_length,
            )
            self.samples.append(
                {
                    "input_ids": aligned_chosen["input_ids"],
                    "attention_mask": aligned_chosen["attention_mask"],
                    "chosen_labels": aligned_chosen["labels"],
                    "rejected_labels": aligned_rejected["labels"],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def collate_fn(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, chosen_labels, rejected_labels = [], [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            chosen_labels.append(x["chosen_labels"] + [-100] * pad_len)
            rejected_labels.append(x["rejected_labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


def sequence_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute average per-token log-probability of a sequence of B/I labels under token classification logits,
    ignoring positions where labels == -100.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    valid_mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0

    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid_mask.float()
    lengths = valid_mask.float().sum(dim=-1).clamp(min=1.0)
    return token_logps.sum(dim=-1) / lengths


class PhoBERTDPOTrainer:
    """
    Direct Preference Optimization (DPO) trainer for PhoBERT Word Segmentation.
    Aligns a policy PhoBERTWordSegmenter against a frozen SFT reference model using
    human/gold vs. negative preference pairs.
    """

    def __init__(
        self,
        base_checkpoint: str = "models/phobert_ws",
        fallback_model: str = "vinai/phobert-base",
        beta: float = 0.5,
        sft_anchor_weight: float = 0.25,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        ckpt = base_checkpoint if os.path.exists(base_checkpoint) else fallback_model
        self.device = get_device(device)
        self.beta = beta
        self.sft_anchor_weight = sft_anchor_weight
        self.max_length = max_length

        self.policy_segmenter = PhoBERTWordSegmenter(
            model_name_or_path=ckpt,
            max_length=max_length,
            device=str(self.device),
        )
        # Create a frozen reference model pi_ref
        self.ref_model = copy.deepcopy(self.policy_segmenter.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train(
        self,
        train_pairs: List[dict],
        val_pairs: Optional[List[dict]] = None,
        epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        save_dir: str = "models/phobert_ws_dpo",
    ) -> Dict[str, List[float]]:
        train_ds = PhoBERTDPODataset(
            preference_pairs=train_pairs,
            tokenizer=self.policy_segmenter.tokenizer,
            label2id=self.policy_segmenter.label2id,
            max_length=self.max_length,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )

        optimizer = AdamW(self.policy_segmenter.model.parameters(), lr=learning_rate, weight_decay=0.01)
        history: Dict[str, List[float]] = {
            "dpo_loss": [],
            "reward_margin": [],
            "pref_acc": [],
            "val_pref_acc": [],
        }

        for epoch in range(1, epochs + 1):
            self.policy_segmenter.model.train()
            total_loss = 0.0
            total_margin = 0.0
            total_acc = 0.0
            steps = 0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                policy_out = self.policy_segmenter.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=chosen_labels,
                )
                policy_logits = policy_out.logits
                sft_loss = policy_out.loss

                with torch.no_grad():
                    ref_logits = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits

                policy_chosen_logps = sequence_log_probs_from_logits(policy_logits, chosen_labels)
                policy_rejected_logps = sequence_log_probs_from_logits(policy_logits, rejected_labels)
                ref_chosen_logps = sequence_log_probs_from_logits(ref_logits, chosen_labels)
                ref_rejected_logps = sequence_log_probs_from_logits(ref_logits, rejected_labels)

                dpo_loss, margin, acc = compute_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                    beta=self.beta,
                )

                combined_loss = dpo_loss + self.sft_anchor_weight * sft_loss
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_segmenter.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(dpo_loss.item())
                total_margin += float(margin.item())
                total_acc += float(acc.item())
                steps += 1

            avg_loss = total_loss / max(1, steps)
            avg_margin = total_margin / max(1, steps)
            avg_acc = total_acc / max(1, steps)

            history["dpo_loss"].append(avg_loss)
            history["reward_margin"].append(avg_margin)
            history["pref_acc"].append(avg_acc)

            val_str = ""
            if val_pairs:
                val_acc = self.evaluate_preference_accuracy(val_pairs, batch_size=batch_size)
                history["val_pref_acc"].append(val_acc)
                val_str = f" | val_pref_acc={val_acc:.4f}"

            print(
                f"DPO Epoch {epoch}/{epochs} | dpo_loss={avg_loss:.4f} | "
                f"reward_margin={avg_margin:+.4f} | train_pref_acc={avg_acc:.4f}{val_str}"
            )

        if save_dir:
            self.policy_segmenter.save(save_dir)

        return history

    def evaluate_preference_accuracy(self, pairs: List[dict], batch_size: int = 16) -> float:
        ds = PhoBERTDPODataset(
            preference_pairs=pairs,
            tokenizer=self.policy_segmenter.tokenizer,
            label2id=self.policy_segmenter.label2id,
            max_length=self.max_length,
        )
        if len(ds) == 0:
            return 0.0
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate_fn)
        self.policy_segmenter.model.eval()
        total_acc = 0.0
        steps = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                policy_logits = self.policy_segmenter.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                ref_logits = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

                _, _, acc = compute_dpo_loss(
                    policy_chosen_logps=sequence_log_probs_from_logits(policy_logits, chosen_labels),
                    policy_rejected_logps=sequence_log_probs_from_logits(policy_logits, rejected_labels),
                    ref_chosen_logps=sequence_log_probs_from_logits(ref_logits, chosen_labels),
                    ref_rejected_logps=sequence_log_probs_from_logits(ref_logits, rejected_labels),
                    beta=self.beta,
                )
                total_acc += float(acc.item())
                steps += 1
        return total_acc / max(1, steps)


def build_llm_preference_memory(
    train_pairs: List[dict],
    save_path: str = "models/llm_dpo_memory.json",
) -> Dict[str, object]:
    """
    Extract phrase-level preference rules (preferred compound joins vs. rejected splits/merges)
    from the HITL-DPO training pairs so that Tier-3 LLM segmenters (Qwen / Llama) can inject
    learned human preferences into both few-shot prompts and local compound resolution.
    """
    preferred_compounds = set()
    rejected_compounds = set()

    for pair in train_pairs:
        chosen_words = set(w.lower() for w in pair.get("chosen", "").split() if "_" in w)
        rejected_words = set(w.lower() for w in pair.get("rejected", "").split() if "_" in w)

        for cw in chosen_words - rejected_words:
            preferred_compounds.add(cw)
        for rw in rejected_words - chosen_words:
            rejected_compounds.add(rw)

    # Remove any conflicts where a compound was preferred in one context and rejected in another
    rejected_compounds -= preferred_compounds

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    memory = {
        "preferred_compounds": sorted(preferred_compounds),
        "rejected_compounds": sorted(rejected_compounds),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    return {
        "preferred_count": len(preferred_compounds),
        "rejected_count": len(rejected_compounds),
        "save_path": save_path,
    }
