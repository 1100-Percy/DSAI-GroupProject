"""Build and execute a notebook containing all experiment Python source inline."""
import ast
import os
import sys

import nbformat
from nbclient import NotebookClient

from .data_io import CONFIG, ROOT, RESULTS, cache_identity, file_sha, write_json


class InlineSource(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.level or node.module == "__future__":
            return None
        return node

    def visit_If(self, node):
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
            return None
        return self.generic_visit(node)


def notebook_source(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = InlineSource().visit(tree)
    if path.name == "data_io.py":
        # Embed configuration and the inlined implementation's fingerprints.
        # Cache validation must work even when the Python package is absent.
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets):
                node.value = ast.parse(repr(CONFIG), mode="eval").body
            if isinstance(node, ast.FunctionDef) and node.name == "cache_identity":
                identity = cache_identity()
                node.body = ast.parse(
                    "return {'phrase_sha256': read_json(RESULTS / 'check.json')['phrase_sha256'], "
                    f"'config_sha256': {identity['config_sha256']!r}, "
                    f"'embeddings_sha256': {identity['embeddings_sha256']!r}, "
                    "'normalizer_sha256': file_sha(DATA / 'code/unified_data.py')}"
                ).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def build_notebook():
    path = ROOT / "notebooks/stage1_embedding_comparison.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        ("data_io", "Immutable training inputs and grouped resampling", "Only allowlisted files are read. Phrase bytes and group boundaries are checked. Configuration and inlined-source fingerprints are embedded, so runtime does not require stage1 Python files."),
        ("baseline_ref", "Sparse reference representation", "The two independently normalized branches are fitted within each training partition."),
        ("embeddings", "Four mean-pooled embeddings", "Fold-local word vectors and frozen pretrained features have separate provenance caches."),
        ("probe", "One nested linear probe", "All representations share the same C grid and grouped inner selection protocol."),
        ("figures", "Figures and experiment report", "Plots use English labels and save their source tables. Cached t-SNE is visualization only."),
        ("build_notebook", "Notebook construction utilities", "The complete builder is included for inspection; it is not invoked recursively."),
        ("run", "Command-line utilities", "The command entry point is included but not executed in the notebook."),
    ]
    cells = [nbformat.v4.new_markdown_cell("# Stage 1: embedding comparison with a fixed linear probe\n\nThis notebook inlines all experiment modules and does not import the stage1 package. The default mode reads existing caches, verifies inputs and results, and regenerates tables and figures. No weights are downloaded or models fitted in cache-only mode. Run from the repository or notebooks directory with the recorded environment."),
             nbformat.v4.new_code_cell("RECOMPUTE = False\nimport os\nassert os.environ.get('PYTHONHASHSEED') == '0', 'Start the kernel with PYTHONHASHSEED=0'\nos.environ['TOKENIZERS_PARALLELISM'] = 'false'\nos.environ['MPLBACKEND'] = 'Agg'\nos.environ['PYTHONDONTWRITEBYTECODE'] = '1'")]
    for name, title, description in sections:
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{description}"))
        cells.append(nbformat.v4.new_code_cell(notebook_source(ROOT / f"stage1/{name}.py")))
    cells.append(nbformat.v4.new_code_cell("import torch\ntorch.manual_seed(42)"))
    cells.extend([
        nbformat.v4.new_markdown_cell("## Execute or validate cached results\n\nRecomputation uses the same protocol as the CLI. The default validates immutable inputs and all cached representations, then regenerates figures without fitting any model."),
        nbformat.v4.new_code_cell("""if RECOMPUTE:
    check_inputs()
    extract_all()
    verify_reproducibility()
else:
    frame, assignments = load_data('notebook')
    load_repeats(frame, assignments)
    load_pretrained()
    scores = pd.read_csv(RESULTS / 'fold_scores.csv', encoding='utf-8')
    tuning = pd.read_csv(RESULTS / 'tuning.csv', encoding='utf-8')
    assert len(scores) == 175 and len(tuning) == 1575
    for name in ['w2v', 'ft']:
        for (repeat, fold), rows in load_repeats(frame, assignments).groupby(['repeat', 'fold']):
            valid = rows.row_index.to_numpy()
            train = np.setdiff1d(np.arange(563), valid)
            feature_path = FEATURES / f'{name}_r{repeat}_f{fold}.npz'
            assert feature_path.exists(), 'Cache-only notebook cannot train word models'
            fold_word_features(name, frame, train, valid, int(repeat), int(fold))
create_figures(cache_only=not RECOMPUTE)
write_reports()
summary = pd.read_csv(RESULTS / 'summary.csv', encoding='utf-8')
display(summary)
display(pd.read_csv(RESULTS / 'key_comparisons.csv', encoding='utf-8'))"""),
        nbformat.v4.new_markdown_cell("## Cached plots\n\nAll five figures below were generated from the experiment tables and cached visualization coordinates."),
        nbformat.v4.new_code_cell("from IPython.display import Image, display\nfor name in ['F1_macro_f1', 'F2_class_f1', 'F3_coverage', 'F4_tsne', 'F5_pairwise_delta']:\n    display(Image(filename=str(FIGURES / f'{name}.png')))"),
    ])
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"name": "stage1", "display_name": "Stage 1 recorded environment", "language": "python"}
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    nbformat.write(notebook, path.open("w", encoding="utf-8"))
    # A private kernelspec ensures the subprocess uses this Python, not a system kernel.
    kernel_dir = ROOT / ".cache/jupyter/kernels/stage1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    write_json(kernel_dir / "kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
               "display_name": "Stage 1", "language": "python"})
    old_jupyter = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(ROOT / ".cache/jupyter") + (os.pathsep + old_jupyter if old_jupyter else "")
    try:
        client = NotebookClient(notebook, timeout=600, kernel_name="stage1", resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    finally:
        if old_jupyter is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = old_jupyter
    nbformat.write(notebook, path.open("w", encoding="utf-8"))
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
    write_json(RESULTS / "notebook_validation.json", {"path": str(path.relative_to(ROOT)),
               "recompute": False, "executed_code_cells": sum(c.cell_type == "code" for c in cells),
               "errors": 0, "sha256": file_sha(path), "python": sys.executable})
    print(f"Notebook executed successfully: {path}", flush=True)
