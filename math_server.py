import logging

from mcp.server import MCPServer


logger = logging.getLogger(__name__)
mcp = MCPServer("Math MCP Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    logger.info("Math add executes a=%s b=%s", a, b)
    return a + b


@mcp.tool()
def percentage_improvement(baseline: float, improved: float) -> float:
    """Compute percentage improvement from a nonzero baseline. Lower values are better."""
    logger.info("Percentage improvement executes baseline=%s improved=%s", baseline, improved)
    return (baseline - improved) / baseline * 100


if __name__ == "__main__":
    mcp.run(transport="stdio")
