import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import typer
from wasabi import msg
from tqdm import tqdm

from spacy_llm.util import assemble

Arg = typer.Argument
Opt = typer.Option

load_dotenv()

def run_pipeline(
        # fmt: off
        input_jsonl: Path = Arg(..., help="Path to the input jsonl file."),
        output_jsonl: Path = Arg(..., help="Path to the output JSONL file."),
        config_path: Path = Arg(..., help="Path to the configuration file to use."),
        examples_path: Optional[Path] = Arg(None, help="Path to the examples file to use (few-shot only)."),
        verbose: bool = Opt(False, "--verbose", "-v", help="Show extra information."),
):
    if not os.getenv("OPENAI_API_KEY", None):
        msg.fail(
            "OPENAI_API_KEY env variable was not found. "
            "Set it by running 'export OPENAI_API_KEY=...' and try again.",
            exits=1,
        )

    msg.text(f"Loading config from {config_path}", show=verbose)

    nlp = assemble(
        config_path,
        overrides={}
        if examples_path is None
        else {"paths.examples": str(examples_path)},
    )

    with input_jsonl.open("r", encoding="utf8") as input_file, output_jsonl.open("w", encoding="utf8") as output_file:
        lines = input_file.readlines()
        for line in tqdm(lines, desc="Labeling feedbacks"):
            data = json.loads(line)
            text = data.get('text', '')
            doc = nlp(text)
            data['categories'] = doc.cats
            output_file.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    typer.run(run_pipeline)
            