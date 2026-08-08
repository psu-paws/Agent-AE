import asyncio
import hydra
from omegaconf import DictConfig
from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import bootstrap_logger

logger = bootstrap_logger()

VIDUR_TASKS = [
    {
        "task_id": "vidur_vscode",
        "question": "In the 2018 VSCode blog post on replit.com, what was the command they clicked on in the last video to remove extra lines?\nYou should follow the format instruction in the request strictly and wrap the final answer in \\boxed{}.",
    },
    {
        "task_id": "vidur_sklearn",
        "question": "In the Scikit-Learn July 2017 changelog, what other predictor base command received a bug fix? Just give the name, not a path.\nYou should follow the format instruction in the request strictly and wrap the final answer in \\boxed{}.",
    },
]

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    async def run():
        main_tool_mgr, sub_tool_mgrs, output_formatter = create_pipeline_components(cfg)
        for task in VIDUR_TASKS:
            print(f"\n{'='*60}\nRunning: {task['task_id']}")
            result, answer, log_path, _ = await execute_task_pipeline(
                cfg=cfg,
                task_id=task['task_id'],
                task_description=task['question'],
                task_file_name="",
                main_agent_tool_manager=main_tool_mgr,
                sub_agent_tool_managers=sub_tool_mgrs,
                output_formatter=output_formatter,
                log_dir="../../logs/vidur_traces",
            )
            print(f"Answer: {answer}")
            print(f"Log: {log_path}")
    asyncio.run(run())

if __name__ == "__main__":
    main()
