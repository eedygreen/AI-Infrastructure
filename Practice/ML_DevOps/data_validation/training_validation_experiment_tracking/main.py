import json

import mlflow
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='.', config_name='config')
def go(config: DictConfig):

    wandb.config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True
    )

    os.environ["WAND_PROJECT"] = config["main"]["project_name"]
    os.environ["WAND_RUN_GROUP"] = config["main"]["experiment_name"]

    # get the path at the root of the Mlflow project
    root_path = hydra.utils.get_original_cwd()

    # Serialize random forest configuration
    model_config = os.path.abspath("random_forest_config.json")

    with open(model_config, "w+") as fp:
        json.dump(dict(config["random_forest"]), fp)

    _ = mlflow.run(
        os.path.join(root_path, "random_forest"),
        "main",
        parameters={
            "train_data": config["data"]["train_data"],
            "model_config": model_config
        },
    )

if __name__ == "__main__":
    go()
    