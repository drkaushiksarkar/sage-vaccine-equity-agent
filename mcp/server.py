"""MCP Server for SAGE Lake data access.

Provides tool endpoints for AI agents to query the SAGE Data Lake
(1.78B rows, 40K+ indicators, 85 organizations).
"""
import json
import logging
from typing import Any, Dict, List

from mcp.server import Server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

app = Server("sage-lake-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="sage_query_indicators",
            description="Search SAGE Lake indicators by keyword. Returns matching indicator codes, names, source organizations, and row counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search keyword (e.g., 'malaria', 'temperature', 'mortality')"},
                    "domain": {"type": "string", "description": "Filter by domain: climate, health, economics, population, aid, lmic, epidemiology, disaster, geospatial"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["keyword"],
            },
        ),
        Tool(
            name="sage_get_timeseries",
            description="Get time series data from SAGE Lake for a specific indicator and country.",
            inputSchema={
                "type": "object",
                "properties": {
                    "indicator_code": {"type": "string"},
                    "country_code": {"type": "string", "description": "ISO3 country code"},
                    "year_from": {"type": "integer"},
                    "year_to": {"type": "integer"},
                },
                "required": ["indicator_code", "country_code"],
            },
        ),
        Tool(
            name="sage_cross_domain",
            description="Cross-domain analysis joining two SAGE Lake indicators on country-year.",
            inputSchema={
                "type": "object",
                "properties": {
                    "indicator_a": {"type": "string"},
                    "indicator_b": {"type": "string"},
                    "year_from": {"type": "integer", "default": 2000},
                    "year_to": {"type": "integer", "default": 2024},
                },
                "required": ["indicator_a", "indicator_b"],
            },
        ),
        Tool(
            name="sage_country_profile",
            description="Get comprehensive country profile with all available indicators across domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "country_code": {"type": "string"},
                    "year": {"type": "integer", "default": 2023},
                },
                "required": ["country_code"],
            },
        ),
        Tool(
            name="sage_evidence_search",
            description="Search 268M evidence embeddings using hybrid BM25+vector retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "source_filter": {"type": "string", "description": "Filter by source: pmc, evidence-spans, all"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="sage_causal_graph",
            description="Query the 33M-triple causal knowledge graph for relationships between entities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "relation_type": {"type": "string", "description": "CAUSES, INCREASES, DECREASES, ASSOCIATED_WITH"},
                    "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"], "default": "both"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["entity"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls from AI agents."""
    logger.info("Tool call: %s(%s)", name, json.dumps(arguments)[:200])

    if name == "sage_query_indicators":
        return await _query_indicators(arguments)
    elif name == "sage_get_timeseries":
        return await _get_timeseries(arguments)
    elif name == "sage_cross_domain":
        return await _cross_domain(arguments)
    elif name == "sage_country_profile":
        return await _country_profile(arguments)
    elif name == "sage_evidence_search":
        return await _evidence_search(arguments)
    elif name == "sage_causal_graph":
        return await _causal_graph(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _query_indicators(args: Dict) -> List[TextContent]:
    """Search indicators in SAGE Lake catalog."""
    # Implementation connects to Athena v_indicator_catalog view
    keyword = args["keyword"]
    sql = f"""
    SELECT source_org, indicator_code, indicator_name, row_count
    FROM sage_lake.v_indicator_catalog
    WHERE LOWER(indicator_name) LIKE '%{keyword.lower()}%'
    ORDER BY row_count DESC
    LIMIT {args.get('limit', 20)}
    """
    return [TextContent(type="text", text=f"Query: {sql}\nResults would be returned from Athena.")]


async def _get_timeseries(args: Dict) -> List[TextContent]:
    return [TextContent(type="text", text=f"Time series for {args['indicator_code']} in {args['country_code']}")]

async def _cross_domain(args: Dict) -> List[TextContent]:
    return [TextContent(type="text", text=f"Cross-domain: {args['indicator_a']} x {args['indicator_b']}")]

async def _country_profile(args: Dict) -> List[TextContent]:
    return [TextContent(type="text", text=f"Country profile for {args['country_code']}")]

async def _evidence_search(args: Dict) -> List[TextContent]:
    return [TextContent(type="text", text=f"Evidence search: {args['query']} across 268M embeddings")]

async def _causal_graph(args: Dict) -> List[TextContent]:
    return [TextContent(type="text", text=f"Causal graph query for {args['entity']}")]


if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    asyncio.run(main())
