# Databricks notebook source
# MAGIC %md
# MAGIC # Upload `data/` to Unity Catalog + create a Genie Space
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads every CSV in the repo's `data/` directory
# MAGIC 2. Writes each one as a managed Delta table in `<catalog>.<schema>`
# MAGIC 3. Creates a Databricks Genie Space pointed at exactly those tables
# MAGIC
# MAGIC The Genie creation logic follows the pattern from
# MAGIC https://github.com/chandhana20/reusable-ip-ai-genie (Genie Spaces DAB).
# MAGIC That repo round-trips an existing space via `serialized_space`; here we
# MAGIC build a fresh `serialized_space` payload from the table list.

# COMMAND ----------

# MAGIC %pip install -U -q "databricks-sdk>=0.40.0" requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Target UC catalog")
dbutils.widgets.text("schema", "agent_framework_demo", "Target UC schema")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse id (for Genie)")
dbutils.widgets.text("data_dir", "./data", "Local data dir with CSVs")
dbutils.widgets.text("genie_title", "Agent Framework Demo", "Genie space title")
dbutils.widgets.text(
    "genie_description",
    "Genie space over the agent-framework-lakebase demo tables.",
    "Genie space description",
)
dbutils.widgets.text("genie_parent_path", "", "Genie parent workspace dir (blank = user home)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
DATA_DIR = dbutils.widgets.get("data_dir").strip()
GENIE_TITLE = dbutils.widgets.get("genie_title").strip()
GENIE_DESCRIPTION = dbutils.widgets.get("genie_description").strip()
GENIE_PARENT_PATH = dbutils.widgets.get("genie_parent_path").strip()

assert CATALOG, "catalog is required"
assert WAREHOUSE_ID, "warehouse_id is required"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify catalog exists, then create schema
# MAGIC
# MAGIC On **Default Storage** accounts, `CREATE CATALOG` from SQL fails because UC
# MAGIC has no managed location to infer. Create the catalog once via the UI
# MAGIC (Catalog Explorer → Create catalog), then re-run.

# COMMAND ----------

existing = {r.catalog for r in spark.sql("SHOW CATALOGS").collect()}
if CATALOG not in existing:
    raise RuntimeError(
        f"Catalog '{CATALOG}' does not exist. Create it via Catalog Explorer "
        f"(this account uses Default Storage, so CREATE CATALOG from SQL fails "
        f"without an explicit MANAGED LOCATION), then re-run this cell."
    )

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
print(f"Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Upload each CSV as a managed Delta table

# COMMAND ----------

import os
import re
from pathlib import Path


def _table_name_from_csv(csv_path: str) -> str:
    stem = Path(csv_path).stem.lower()
    return re.sub(r"[^a-z0-9_]", "_", stem)


def _load_csv_to_uc(csv_path: str, catalog: str, schema: str) -> str:
    table = _table_name_from_csv(csv_path)
    fqn = f"`{catalog}`.`{schema}`.`{table}`"

    # Spark can't read directly from the driver's local fs on serverless; copy to a volume-like
    # path via dbutils.fs if needed. For repo-relative paths in a Databricks Git Folder this
    # `file:` URI works on classic clusters; for serverless, set DATA_DIR to a Volumes path.
    abs_path = os.path.abspath(csv_path)
    read_uri = abs_path if abs_path.startswith("/Volumes/") else f"file:{abs_path}"

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(read_uri)
    )
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn)
    print(f"  wrote {fqn}  ({df.count():,} rows, {len(df.columns)} cols)")
    return f"{catalog}.{schema}.{table}"


csv_files = sorted(str(p) for p in Path(DATA_DIR).glob("*.csv"))
assert csv_files, f"no CSVs found in {DATA_DIR}"
print(f"Found {len(csv_files)} CSVs:")
for p in csv_files:
    print(f"  {p}")

uploaded_tables = [_load_csv_to_uc(p, CATALOG, SCHEMA) for p in csv_files]
print("\nUploaded:")
for t in uploaded_tables:
    print(f"  {t}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Genie Space pointed at exactly these tables
# MAGIC
# MAGIC Uses `POST /api/2.0/genie/spaces` with a `serialized_space` payload —
# MAGIC the same shape the reusable-ip-ai-genie DAB writes when it round-trips
# MAGIC a space, just constructed from scratch here.

# COMMAND ----------

# DBTITLE 1,Create Genie Space from UC schema
import json
import uuid
import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
GENIE_SPACES_API = "api/2.0/genie/spaces"


class GenieApiError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=["GET", "POST", "PATCH"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _ctx():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext()


def _host() -> str:
    return _ctx().apiUrl().get().rstrip("/")


def _headers() -> dict:
    return {"Authorization": f"Bearer {_ctx().apiToken().get()}"}


def _parent_path(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    username = _ctx().userName().get()
    return f"/Users/{username}"


def _build_serialized_space(tables: list[str]) -> str:
    """Build the serialized_space JSON required by the Genie Spaces API.

    Args:
        tables: List of fully-qualified table names (catalog.schema.table).

    Returns:
        A JSON string conforming to the serialized_space v2 schema.
    """
    # Tables must be sorted alphabetically by identifier
    sorted_tables = sorted(tables)
    table_entries = [{"identifier": t} for t in sorted_tables]

    space_obj = {
        "version": 2,
        "config": {
            "sample_questions": []
        },
        "data_sources": {
            "tables": table_entries
        },
        "instructions": {
            "text_instructions": [],
            "example_question_sqls": [],
            "join_specs": [],
        },
    }
    return json.dumps(space_obj)


def create_genie_space_from_schema(
    catalog: str,
    schema: str,
    warehouse_id: str,
    title: str,
    description: str,
    parent_path: Optional[str] = None,
) -> dict:
    """Create a Genie Space pointed at all tables in a UC schema.

    Args:
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        warehouse_id: SQL warehouse ID for the Genie Space.
        title: Display title for the space.
        description: Description for the space.
        parent_path: Workspace directory to create the space in (defaults to user home).

    Returns:
        API response dict containing the new space_id.
    """
    # Discover all tables in the schema
    tables_df = spark.sql(f"SHOW TABLES IN `{catalog}`.`{schema}`")
    table_names = [
        f"{catalog}.{schema}.{row.tableName}" for row in tables_df.collect()
    ]
    if not table_names:
        raise ValueError(f"No tables found in {catalog}.{schema}")

    print(f"Discovered {len(table_names)} tables in {catalog}.{schema}:")
    for t in sorted(table_names):
        print(f"  {t}")

    serialized_space = _build_serialized_space(table_names)
    resolved_parent = _parent_path(parent_path)

    payload = {
        "warehouse_id": warehouse_id,
        "parent_path": resolved_parent,
        "title": title,
        "description": description,
        "serialized_space": serialized_space,
    }

    url = f"{_host()}/{GENIE_SPACES_API}"
    resp = _session().post(url, headers=_headers(), json=payload)
    if not resp.ok:
        raise GenieApiError(
            f"Failed to create Genie Space: {resp.status_code} {resp.text}"
        )
    return resp.json()


# COMMAND ----------

# DBTITLE 1,Run: create Genie Space
result = create_genie_space_from_schema(
    catalog=CATALOG,
    schema=SCHEMA,
    warehouse_id=WAREHOUSE_ID,
    title=GENIE_TITLE,
    description=GENIE_DESCRIPTION,
    parent_path=GENIE_PARENT_PATH or None,
)

space_id = result.get("space_id") or result.get("id")
print(f"\nCreated Genie Space: {space_id}")
print(f"URL: {_host()}/genie/rooms/{space_id}")
print(f"\nPut this in agent_graph_walkthrough_public.py:")
print(f'  GENIE_SPACE_ID = "{space_id}"')

# COMMAND ----------

