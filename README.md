# Agent Framework + Lakebase Demo

End-to-end multi-agent system on Databricks: LangGraph supervisor, Lakebase
chat memory, Genie + SQL + HTTP worker agents, MLflow tracing.

## Prereqs

- Databricks workspace on **AWS or Azure**, Unity Catalog enabled
- A **SQL warehouse** (serverless preferred) — note its ID
- An **existing UC catalog** you can write to (on Default Storage accounts you must create it via Catalog Explorer, not from SQL)
- Permission to **create a Lakebase instance** and a **Genie space**
- Access to the `databricks-claude-sonnet-4-6` Foundation Model endpoint
- Compute for the notebook: **Serverless** or a classic cluster on **DBR 15.4 ML+**

## 1. Clone into the workspace

Use Git Folders: **Workspace → Repos → Add Repo** → this repo's URL.

## 2. Load `data/` into Unity Catalog + create the Genie space

Open `upload_data_and_create_genie.py` as a notebook, attach compute, fill in
the widgets at the top:

| Widget | Value |
|---|---|
| `catalog` | your target UC catalog |
| `schema` | e.g. `agent_framework_demo` |
| `warehouse_id` | SQL warehouse ID |
| `data_dir` | `./data` on classic, or a `/Volumes/...` path on serverless |

Run all cells. The last cell prints the new **Genie Space ID** and URL — copy
the ID.

## 3. Configure and run the demo notebook

Open `agent_graph_walkthrough_public.py` and edit the constants in section 1:

```python
GENIE_SPACE_ID = "<from step 2>"
WAREHOUSE_ID   = "<your warehouse id>"
CATALOG        = "<same catalog as step 2>"
USER_EMAIL     = "you@yourcompany.com"
```

Run cell-by-cell the first time (Lakebase provisioning takes 2–5 min on first
create). After that, **Run all** takes ~3 minutes.
