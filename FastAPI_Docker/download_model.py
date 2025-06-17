import os
import wandb

wandb.require("core")

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

run = wandb.init(project="linear-regression-pytorch", entity="s-oksana-set-university")

artifact = run.use_artifact('s-oksana-set-university/linear-regression-pytorch/linear_regression_model:v0', type='model')
path = artifact.get_path("linear_regression_model.pth")  

downloaded_path = path.download('./model/')
print("✅ Модель збережено за шляхом:", downloaded_path)

