import json
import logging
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel


logger = logging.getLogger(__name__)
mcp = MCPServer("Research MCP Server")
EXPERIMENTS_PATH = Path(__file__).resolve().parent / "data" / "experiments.json"


class Experiment(BaseModel):
    run_id: str
    algorithm: str
    benchmark: str
    dimension: int
    final_fitness: float
    note: str


def load_experiments() -> list[Experiment]:
    """Load experiment records from the local JSON file."""
    logger.info("Loading experiments from %s", EXPERIMENTS_PATH)
    with EXPERIMENTS_PATH.open(encoding="utf-8") as file:
        raw_data = json.load(file)
    return [Experiment(**item) for item in raw_data]


@mcp.tool(
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def search_experiments(query: str) -> list[Experiment]:
    """Find experiments by case-insensitive substring in run ID, algorithm, benchmark, or note."""
    query = query.strip().casefold()
    if not query:
        message = "Search query must not be empty."
        logger.warning(message)
        raise ToolError(message)

    matches = []
    for experiment in load_experiments():
        fields = (experiment.run_id, experiment.algorithm, experiment.benchmark, experiment.note)
        if any(query in field.casefold() for field in fields):
            matches.append(experiment)
    return matches


@mcp.resource("research://project/overview", mime_type="text/markdown")
def get_project_overview() -> str:
    """Overview of the research project and its experiment data."""
    return (
        "# Research Project\n\n"
        "This project provides access to local optimization experiment records.\n\n"
        "## Data\n"
        "- Each record includes an algorithm, benchmark, dimension, final fitness, and note.\n\n"
        "## MCP Capabilities\n"
        "- Search records with `search_experiments`.\n"
        "- Read a record at `research://experiments/{run_id}`.\n"
    )


@mcp.resource("research://experiments/{run_id}", mime_type="application/json")
def get_experiment(run_id: str) -> Experiment:
    """Read an experiment record by its run ID."""
    for experiment in load_experiments():
        if experiment.run_id == run_id:
            return experiment
    raise ResourceNotFoundError(f"Experiment {run_id!r} not found.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
