from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import typer
from rich import print
from ptliq.service.scoring import Scorer

app = typer.Typer(no_args_is_help=True)

def _read_records(path: Path) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() in [".jsonl", ".ndjson"]:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif p.suffix.lower() in [".json"]:
        rows = json.loads(p.read_text())
        if isinstance(rows, dict) and "rows" in rows:
            rows = rows["rows"]
    elif p.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(p)
        rows = df.to_dict(orient="records")
    else:
        raise typer.BadParameter(f"Unsupported input type: {p.suffix}")
    return rows

@app.command()
def app_main(
    package: Path = typer.Option(..., help="model dir or model zip"),
    input_path: Path = typer.Option(..., help="parquet | json | jsonl with f_* fields"),
    output_path: Path = typer.Option(..., help="parquet|jsonl for predictions"),
    device: str = typer.Option("cpu"),
):
    """
    Score a batch of rows offline. Rows must contain f_* features; missing values are imputed to the scaler mean.
    """
    scorer = Scorer.from_dir(package, device=device) if Path(package).is_dir() else Scorer.from_zip(package, device=device)
    rows = _read_records(input_path)
    y = scorer.score_many(rows)

    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        import pandas as pd
        pd.DataFrame({"preds_bps": y}).to_parquet(out, index=False)
    else:
        with open(out, "w", encoding="utf-8") as f:
            for v in y:
                f.write(json.dumps({"preds_bps": float(v)}) + "\n")

    print(f"[bold green]SCORED[/bold green] {len(y)} rows → {out}")

app = app

if __name__ == "__main__":
    app()
