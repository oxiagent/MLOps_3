import os
import torch
import wandb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from model_def import LinearRegressionModel  # 🔹 Імпорт класу твоєї моделі

# FastAPI додаток
app = FastAPI()

# 🔹 Pydantic-схема для вводу фічей
class Features(BaseModel):
    features: list[float]


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle) -> None:
        self.handle: DeploymentHandle = object_detection_handle.options(
            use_new_handle_api=True
        )

    @app.post("/predict")
    async def predict(self, features: Features):
        result = await self.handle.predict.remote(features.features)
        return JSONResponse(content=result)


@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
    ray_actor_options={"num_cpus": 1}
)
class ObjectDetection:
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
            state_dict = torch.load(model_file, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ Модель успішно завантажена з wandb!")

        except Exception as e:
            print(f"❌ Не вдалося завантажити модель: {e}")
        finally:
            wandb.finish()

    async def predict(self, features: list[float]):
        print(f"🔍 Отримані фічі: {features}")
        input_tensor = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            output = self.model(input_tensor).numpy().tolist()

        return {"status": "ok", "prediction": output}


# Запуск
entrypoint = APIIngress.bind(ObjectDetection.bind())
