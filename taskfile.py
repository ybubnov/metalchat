from pathlib import Path


def prepare_safetensors(
    repo_id: str, root_path: str, weights_path: str, exclude: set[str] | None = None
) -> None:
    import torch
    from safetensors.torch import save_file

    model_path = Path(root_path) / repo_id / weights_path
    model = torch.load(model_path, "cpu", weights_only=True)

    for excl in (exclude or set()):
        del model[excl]

    save_file(model, model_path.parent / "model.safetensors")


def prepare_test_fixture(repo_id: str, root_path: str = "text_fixture") -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=Path(root_path) / repo_id)


def prepare_test_fixtures(root_path: str = "test_fixture") -> None:
    prepare_test_fixture("meta-llama/Llama-3.2-1B-Instruct", root_path)
    prepare_test_fixture("meta-llama/LLama-3.2-1B-Instruct-QLORA_INT4_EO8", root_path)

    prepare_safetensors(
        "meta-llama/Llama-3.2-1B-Instruct",
        root_path,
        weights_path="original/consolidated.00.pth",
        exclude={"output.weight"},
    )

    prepare_safetensors(
        "meta-llama/LLama-3.2-1B-Instruct-QLORA_INT4_EO8",
        root_path,
        weights_path="consolidated.00.pth",
    )
