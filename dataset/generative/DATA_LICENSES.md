# Data in this folder: sources and licences

The committed data files here (`dev/`, `gold/`, `gold2/`, `eval/`, `pilot/`, `round_v3/`, `judge_calibration/`)
hold questions, SQL, result previews and model answers about the tables of the data lake built by `build_lake*.py`.
Values quoted from a table keep that table's licence. The full source list, licences and attribution notices are in
the card of the public dataset [developmentseed/gazet-sql-data](https://huggingface.co/datasets/developmentseed/gazet-sql-data).

Summary of the sources: Open Government Licence v3.0 (UK police, NaPTAN, ONS); ODbL (Paris trees and Vélib',
OpenFlights, Overture buildings, roads and divisions; © OpenStreetMap contributors); CDLA-Permissive-2.0 (Overture
places, including Foursquare data under Apache-2.0); CC BY 4.0 (CBS/PDOK, World Bank, WRI Global Power Plant
Database, GeoNames); CC BY 2.5 India (DataMeet districts); Singapore Open Data Licence v1.0 (NParks); PDL 1.0
(Global Map Japan, GSI); Licence Ouverte 2.0 (IGN/INSEE départements); IBGE, INPE Programa Queimadas, Ecobici GBFS
feeds and NYC TLC data, used with credit; public domain (Natural Earth, US Census Bureau, USGS, NOAA NCEI,
OurAirports).

**Removed:** rows about the EU NUTS regions table (`eu_nuts_regions`, Eurostat GISCO, © EuroGeographics), whose
licence allows non-commercial use only. Since 2026-11-19 the gold and dev sets here exclude them (gold v1 291 of 342
questions, gold v2 137 of 158, dev v1 686 of 755); numbers in older reports were measured on the full sets.
`build_lake_world.py` still downloads the table for local, non-commercial experiments.

Model outputs in `pilot/`, `round_v3/`, `judge_calibration/` and `eval/` come from the models named in each file
(DeepSeek, GLM, MiniMax, Qwen, Mistral, gpt-oss, Claude Haiku, Muse Spark Open and the gazet models); they are
evaluation records, and no model published by this project was trained on the Claude Haiku or Muse Spark Open outputs.
