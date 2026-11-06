from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceSoftValueEstimator:
    """
    Local backend exposing Blondel's ARM-side soft value V_q(s).

    This reads the one-step log-partition from the next-token logits and is
    intended for branch decisions, not for verifier-weighted PLP rollouts.
    """

    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
        )
        self.model.eval()

        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def estimate_v_s(
        self,
        prompt: str,
        temperature: float = 1.0,
        gauge_fix: str = "none",
    ) -> float:
        """
        Estimatest the continuation free energy V_q(s) from the logits
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # for qwen models this is useful to avoid 200+ useless tokens used for padding to a power of 2
        vocab_size = self.model.config.vocab_size
        next_token_logits = outputs.logits[0, -1, :vocab_size].float()

        if gauge_fix == "none":
            adjusted_logits = next_token_logits
        elif gauge_fix == "mean":
            adjusted_logits = next_token_logits - next_token_logits.mean()
        elif gauge_fix == "z_score":
            mean = next_token_logits.mean()
            std = next_token_logits.std()
            adjusted_logits = (next_token_logits - mean) / (std + 1e-8)
        else:
            raise ValueError("gauge_fix must be either 'none' or 'mean' or 'z_score'")

        scaled_logits = adjusted_logits / temperature
        value = temperature * torch.logsumexp(scaled_logits, dim=-1)
        return value.item()

    def score_option(self, prompt: str, option: str) -> float:
        """
        Scores an option using length-normalized log-likelihood (EBM style)

        ```
        \text{Score}(y|x) = \frac{1}{|y|} \\sum_{t=1}^{|y|} \big( q(s_t, y_t) - V_q(s_t) \big)
        ```
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        option_ids = self.tokenizer(
            option, return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        input_ids = torch.cat([prompt_ids, option_ids], dim=-1).to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[0]

        # We only care about the transitions during the option
        start_idx = prompt_ids.shape[1] - 1
        end_idx = start_idx + option_ids.shape[1]

        # q(s_t, y_t)
        target_logits = logits[start_idx:end_idx, :]
        actual_token_ids = option_ids[0]
        q_s_y = target_logits[torch.arange(len(actual_token_ids)), actual_token_ids]

        # V_q(s_t)
        V_q = torch.logsumexp(target_logits, dim=-1)

        # Log likelihood is exactly q(s,y) - V_q(s)
        log_probs = q_s_y - V_q

        # Length normalized Likelihood Ratio
        return log_probs.mean().item()
