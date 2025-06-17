import ray
from ray import serve
import os
from dotenv import load_dotenv

# Завантажуємо змінні середовища з файлу .env
load_dotenv()

# Ініціалізація Ray
ray.init(
    address="ray://localhost:10001",
    runtime_env={
        "working_dir": ".",
        "pip": [
            "wandb",
            "python-dotenv",
            "torch",
            "fastapi",
            "uvicorn",
            "pydantic",
            "requests"
        ],
        "env_vars": {
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "linear-regression-pytorch"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY", "s-oksana-set-university"),
            "WANDB_MODEL_ARTIFACT": os.getenv(
                "WANDB_MODEL_ARTIFACT",
                "s-oksana-set-university/linear-regression-pytorch/linear_regression_model:v0"
            ),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true"
        }
    }
)

# Імпорт entrypoint з model_def.py
from model_def import entrypoint

# Запуск моделі
serve.run(entrypoint, name="linear-regression")
