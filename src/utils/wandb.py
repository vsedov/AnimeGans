import wandb
from loguru import logger as log

log.info("Starting wandb")


def init_wandb(project_name, run_name, config):
    """Initialize wandb"""
    wandb.init(project=project_name, name=run_name, config=config)


def log_wandb(loss, metrics, epoch):
    """Log metrics to wandb"""
    wandb.log({"loss": loss, "metrics": metrics, "epoch": epoch})


def finish_wandb():
    """Finish wandb run"""
    wandb.finish()


def get_best_model(project_id, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"vsedov/{project_id}/{sweep_id}")
    runs = sorted(
        sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True
    )
    val_acc = runs[0].summary.get("val_acc", 0)
    log.info(f"Best model has val_acc={val_acc}")
    #  REVISIT: (vsedov) (17:01:50 - 25/11/22): This might have to be changed im not sure
    runs[0].file("mode.h5").download(replace=True)
