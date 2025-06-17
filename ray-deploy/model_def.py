import os
import torch
import wandb
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from torch import nn

# FastAPI app
app = FastAPI()

# Pydantic модель для запиту
class Features(BaseModel):
    features: list[float]

# Клас моделі
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Ingress endpoint
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle) -> None:
        self.handle: DeploymentHandle = model_handle.options(use_new_handle_api=True)

    @app.post("/predict")
    async def predict(self, features: Features):
        result = await self.handle.predict.remote(features.features)
        return JSONResponse(content=result)

# Модель
@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
    ray_actor_options={"num_cpus": 1}
)
class LinearRegressionModelDeployment:
    def __init__(self):
        self.wandb_project = os.getenv("WANDB_PROJECT", "linear-regression-pytorch")
        self.wandb_entity = os.getenv("WANDB_ENTITY", "s-oksana-set-university")
        self.model_artifact_name = os.getenv(
            "WANDB_MODEL_ARTIFACT",
            "s-oksana-set-university/linear-regression-pytorch/linear_regression_model:v0"
        )

        print("🤖 Ініціалізація wandb та завантаження моделі...")
        os.environ["WANDB_MODE"] = "online"
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            job_type="inference",
            mode="online"
        )

        try:
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key:
                raise ValueError("WANDB_API_KEY not found in environment variables")

            print(f"📥 Завантаження артефакту моделі: {self.model_artifact_name}")
            artifact = run.use_artifact(self.model_artifact_name, type='model')
            model_path = artifact.download()

            model_file = None
            for file in os.listdir(model_path):
                if file.endswith('.pt') or file.endswith('.pth'):
                    model_file = os.path.join(model_path, file)
                    break

            if model_file is None:
                raise FileNotFoundError("No .pt or .pth model file found in the downloaded artifact")

            print(f"📁 Шлях до файлу моделі: {model_file}")
            self.model = LinearRegressionModel()
            self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            self.model.eval()
            print("✅ Модель успішно завантажена з wandb!")

        except Exception as e:
            print(f"❌ Не вдалося завантажити модель: {e}")
        finally:
            wandb.finish()

    async def predict(self, features: list[float]):
        input_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor).numpy().tolist()
        return {"status": "ok", "prediction": output}

# Entry point для serve
entrypoint = APIIngress.bind(LinearRegressionModelDeployment.bind())
