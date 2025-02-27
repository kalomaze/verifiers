from trl import GRPOTrainer
import verifiers as vf
import wandb
import os
import datetime
import shutil
import glob

model_name = "Quest-AI/quest-corruption-7b-s330-v3"
model, tokenizer = vf.get_model_and_tokenizer(
    "Quest-AI/quest-corruption-7b-s330-v3",
    use_chat_template=False
)

# Create environment
vf_env = vf.CorruptionEnv(
    policy_model=model,
    tokenizer=tokenizer,
    dataset="quest-corruption"
)


dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(
    run_name="q-corruption-2.5-7b-ematest",
    num_gpus=8
)

if int(os.getenv("RANK", 0)) == 0:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    copy_dir = f"run_files_{timestamp}"
    os.makedirs(copy_dir, exist_ok=True)

    files_to_copy = ["*.py", "verifiers/**/*.py"]
    for pattern in files_to_copy:
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.isfile(filepath):
                dest_path = os.path.join(copy_dir, os.path.relpath(filepath))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(filepath, dest_path)
    wandb.init()
    wandb.save(os.path.join(copy_dir, "*.py"))
    wandb.save(os.path.join(copy_dir, "verifiers/**/*.py"))

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

if int(os.getenv("RANK", 0)) == 0:
    wandb.finish()