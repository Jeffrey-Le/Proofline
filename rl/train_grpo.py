"""Train a small open model on GroundedMatch with GRPO (TRL) + LoRA.

    pip install -r rl/requirements.txt
    python -m rl.train_grpo --model Qwen/Qwen2.5-1.5B-Instruct --steps 300 --out runs/gm-qwen1.5b

Sized for one 24-40 GB GPU (a Colab A100, a lab GPU). Generation dominates the time;
add --use-vllm on a machine with vLLM installed for a large speedup.
Training data is synthetic only (rl/synth.py): no real resumes are ever used.
"""
import argparse

from .env import format_reward, grounded_reward, render
from .synth import make_split


def build_dataset(n: int, start: int = 0):
    from datasets import Dataset
    eps = make_split(n, "train", start=start)
    return Dataset.from_list([{"prompt": render(e), "episode": e.to_json()} for e in eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02, help="KL penalty toward the base model")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--out", default="runs/grounded-match")
    a = ap.parse_args()

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    args = GRPOConfig(
        output_dir=a.out,
        max_steps=a.steps,
        learning_rate=a.lr,
        beta=a.beta,
        num_generations=a.num_generations,
        per_device_train_batch_size=a.num_generations,  # one prompt's group per device step
        gradient_accumulation_steps=4,
        max_completion_length=320,
        temperature=1.0,
        reward_weights=[1.0, 1.0],                      # grounded_reward, format_reward
        logging_steps=5,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        log_completions=True,
        use_vllm=a.use_vllm,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=[grounded_reward, format_reward],
        args=args,
        train_dataset=build_dataset(a.train_size),
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(a.out)
    print(f"saved LoRA adapter to {a.out}; evaluate with: python -m rl.evaluate --policy hf:{a.out} --base {a.model}")


if __name__ == "__main__":
    main()
