from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root (config.py lives at backend/app/config.py) — anchored here so the
# default data dir does not depend on the process CWD
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YVT_", env_file=".env", extra="ignore")

    data_dir: Path = _REPO_ROOT / "data"
    # "auto" resolves to cuda > mps > cpu at inference time
    device: str = "auto"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.models_dir,
            self.projects_dir,
            self.jobs_dir,
            self.runs_dir,
            self.datasets_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()


def resolve_device(device: str | None = None) -> str:
    dev = device or settings.device
    if dev != "auto":
        return dev
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cpu_brand() -> str:
    """Best-effort human-readable CPU/chip name (e.g. 'Apple M2')."""
    import platform

    if platform.system() == "Darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def get_device_info(device: str | None = None) -> dict:
    """Resolved compute device + accelerator details for the /api/system endpoint.

    VRAM figures are only available for CUDA; MPS does not expose them via torch.
    """
    import platform

    import torch

    resolved = resolve_device(device)
    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )

    info: dict = {
        "device": resolved,
        "requested": device or settings.device,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "vram_total_mb": None,
        "vram_used_mb": None,
    }

    if resolved not in ("cpu", "mps") and cuda_available:
        try:
            idx = int(resolved)
        except ValueError:
            idx = 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info["accelerator"] = "cuda"
        info["accelerated"] = True
        info["device_name"] = props.name
        info["vram_total_mb"] = round(total / 1024 / 1024)
        info["vram_used_mb"] = round((total - free) / 1024 / 1024)
        info["device_label"] = f"{props.name} (CUDA)"
    elif resolved == "mps":
        chip = _cpu_brand()
        info["accelerator"] = "mps"
        info["accelerated"] = True
        info["device_name"] = chip
        info["device_label"] = f"{chip} (MPS)"
    else:
        chip = _cpu_brand()
        info["accelerator"] = "cpu"
        info["accelerated"] = False
        info["device_name"] = chip
        info["device_label"] = f"{chip} (CPU)"

    return info
