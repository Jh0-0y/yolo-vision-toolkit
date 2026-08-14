from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from lib import device as device_lib

# repo root (config.py lives at backend/app/core/config.py) — anchored here so
# the default data dir does not depend on the process CWD
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YVT_", env_file=".env", extra="ignore")

    data_dir: Path = _REPO_ROOT / "data"
    # optional fast scratch (e.g. an SSD mount) to stage the active dataset onto
    # before training — copied in when a run starts, removed when it ends. None
    # disables staging entirely: training reads straight from datasets_dir,
    # wherever data_dir lives. Set YVT_SSD_CACHE_DIR to enable on servers whose
    # bulk storage is a slow HDD.
    ssd_cache_dir: Path | None = None
    # "auto" resolves to cuda > mps > cpu at inference time
    device: str = "auto"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    # single source of truth for the versioned API mount point (routers + emitted URLs)
    api_prefix: str = "/api/v1"

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

    @property
    def labs_dir(self) -> Path:
        """연구실(Lab) 공간. 학습실 프로젝트와 나란한 최상위 자리다 —
        연구실은 라벨도 클래스도 쓰지 않아 프로젝트에 속하지 않는다."""
        return self.data_dir / "labs"

    def lab_dir(self, lab_id: str) -> Path:
        return self.labs_dir / lab_id

    @property
    def test_dir(self) -> Path:
        """Scratch space for the Test playground (uploads, video frames).
        Nothing here is committed to the DB — it's transient."""
        return self.data_dir / "test"

    def model_dir(self, project_id: str | None, model_id: str) -> Path:
        """Where a model's files live. Scoped models sit under their project
        folder; shared/legacy models (project_id None) stay in the flat pool."""
        if project_id:
            return self.projects_dir / project_id / "models" / model_id
        return self.models_dir / model_id

    def run_dir(self, project_id: str | None, run_id: str) -> Path:
        """Where a training run's artifacts live. Scoped runs sit under their
        project folder; shared/legacy runs stay in the flat pool."""
        if project_id:
            return self.projects_dir / project_id / "runs" / run_id
        return self.runs_dir / run_id

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.models_dir,
            self.projects_dir,
            self.jobs_dir,
            self.runs_dir,
            self.datasets_dir,
            self.labs_dir,
            self.test_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def get_resource_info(device: str | None = None) -> dict:
    """Device info + live memory pressure, for the Test page's resource guard.

    CUDA reports VRAM (via torch); MPS/CPU report system RAM (unified memory on
    Apple = the relevant budget). psutil is cross-platform (Windows + macOS).
    """
    import psutil

    info = get_device_info(device)
    vm = psutil.virtual_memory()
    info["ram_total_mb"] = round(vm.total / 1024 / 1024)
    info["ram_available_mb"] = round(vm.available / 1024 / 1024)
    info["ram_used_mb"] = round((vm.total - vm.available) / 1024 / 1024)
    if info.get("vram_total_mb") and info.get("vram_used_mb") is not None:
        info["vram_free_mb"] = info["vram_total_mb"] - info["vram_used_mb"]
    return info


settings = Settings()


def resolve_device(device: str | None = None) -> str:
    """요청 디바이스를 실제 장치 문자열로 해석한다.

    앱이 하는 일은 **기본값을 채우는 것**뿐이고(요청에 값이 없으면 `settings.device`),
    "auto" 를 무엇으로 볼지는 `lib.device` 가 정한다. 순수 계산 쪽은 이 함수를
    부르지 않는다 — 이미 해석된 문자열을 인자로 받는다.
    """
    return device_lib.resolve(device or settings.device)


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
