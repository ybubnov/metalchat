from pathlib import Path


def prepare_llama(repo_id: str, path: str, subdir: str = "", no_output: bool = True) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    for remote_filename in ["consolidated.00.pth", "params.json", "tokenizer.model"]:
        hf_hub_download(
            repo_id=repo_id,
            filename=str(Path(subdir) / remote_filename),
            local_dir=path,
            force_download=False,
        )

    model_path = Path(path) / Path(subdir) / "consolidated.00.pth"
    model = torch.load(model_path, "cpu", weights_only=True)
    if no_output:
        del model["output.weight"]

    save_file(model, Path(path) / "model.safetensors")


def prepare_test_fixture(path: str = "test_fixture") -> None:
    prepare_llama(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        path=str(Path(path) / "llama3.2:1b-instruct"),
        subdir="original"
    )

    prepare_llama(
        repo_id="meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
        path=str(Path(path) / "llama3.2:1b-qlora"),
        no_output=False,
    )
