# Sage Vaccine Equity Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![MCP](https://img.shields.io/badge/MCP-1.0-green) ![n8n](https://img.shields.io/badge/n8n-automation-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

Agentic AI platform with MCP tool servers for vaccine coverage equity analysis using UNICEF (8.8M rows), WHO immunization, and OECD ODA financing data

## Architecture

```
SAGE Data Lake (1.78B rows)     Processing Layer              Application Layer
+-------------------------+    +----------------------+    +-------------------+
| ERA5 Climate (362M)     |    | Feature Engineering  |    | REST API          |
| WHO Health (290M)       |--->| Model Training/Infer |--->| Real-time Stream  |
| IHME GBD (47M)          |    | Knowledge Graph      |    | Dashboard UI      |
| 268M Vector Embeddings  |    | Agent Orchestration  |    | Alert System      |
| 33M Causal KG Triples   |    | Evaluation Pipeline  |    | Report Generator  |
+-------------------------+    +----------------------+    +-------------------+
```

## Data Sources (SAGE Lake)

| Source | Records | Domain | Usage |
|--------|---------|--------|-------|
| WHO GHO | 190M+ | Health | Disease indicators across 194 countries |
| ERA5 CCD | 362M+ | Climate | Daily climate reanalysis (temperature, precipitation) |
| IHME GBD | 47M+ | Health | Global burden of disease estimates |
| World Bank WDI | 24M+ | Economics | Socioeconomic covariates |
| UNICEF | 8.8M+ | Child Health | Immunization, nutrition, WASH indicators |
| OpenDengue | 5.7M+ | Epidemiology | Dengue case counts across 129 countries |
| OECD Pharma | 19.6M+ | Pharmaceutical | Drug consumption, pricing, expenditure |
| CARD AMR | 271K | Genomics | Antimicrobial resistance genomes |
| WorldPop COGs | 200K+ | Geospatial | Population density rasters at 100m/1km |

## Quick Start

```bash
# Backend
pip install -r requirements.txt
python -m src.main

# Frontend (if applicable)
cd frontend && npm install && npm run dev
```

## License

MIT License -- Kaushik Sarkar 2025
