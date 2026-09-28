"""Executa o notebook do inicio ao fim, gravando os outputs no proprio arquivo."""
import sys, time, nbformat
from pathlib import Path
from nbclient import NotebookClient

caminho = Path(__file__).resolve().parents[1] / "metacaracteristicas_dominio.ipynb"
nb = nbformat.read(caminho, as_version=4)
inicio = time.perf_counter()
cliente = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(caminho.parent)}})
try:
    cliente.execute()
finally:
    nbformat.write(nb, caminho)
erros = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and any(o.output_type == "error" for o in c.outputs)]
print(f"executado em {time.perf_counter() - inicio:.0f}s | erros nas celulas: {erros or 'nenhuma'}", flush=True)
sys.exit(1 if erros else 0)
