import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "RehabLLM_v04_Clean_Enrich_SFT.ipynb"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_v04_notebook_code_cells_compile():
    notebook = _load_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_v04_document_target_is_feasible_from_configured_source_caps():
    notebook = _load_notebook()
    settings = next(
        "".join(cell.get("source") or [])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "V04_TARGET_DOCS" in "".join(cell.get("source") or [])
    )
    tree = ast.parse(settings)
    values = {}
    wanted = {"BASE_TARGET_DOCS", "V04_TARGET_DOCS", "ROBOTICS_MAX_ARTICLES"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)

    assert wanted <= values.keys()
    assert values["V04_TARGET_DOCS"] <= (
        values["BASE_TARGET_DOCS"] + values["ROBOTICS_MAX_ARTICLES"]
    )


def test_v04_notebook_contains_gpu_readiness_gate():
    notebook = _load_notebook()
    all_source = "\n".join(
        "".join(cell.get("source") or []) for cell in notebook["cells"]
    )
    assert "assess_v04_readiness.py" in all_source
    assert "READY FOR GPU" in all_source
