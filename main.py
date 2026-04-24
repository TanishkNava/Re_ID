import uvicorn
import yaml

if __name__ == "__main__":
    with open("configs/pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)["api"]
    uvicorn.run("api.server:app", host=cfg["host"], port=cfg["port"], reload=False)