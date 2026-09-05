from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_lock_contains_opencv_provider_for_cv2():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "opencv-python==" in lock


def test_linux_accelerator_packages_are_platform_guarded():
    lines = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    binary_prefixes = ("cuda-", "nvidia-c", "nvidia-n", "triton==")
    accelerator_lines = [line for line in lines if line.startswith(binary_prefixes)]
    assert accelerator_lines
    assert all("sys_platform == 'linux'" in line for line in accelerator_lines)


def test_locks_are_generated_as_universal_resolutions():
    for filename in ("requirements.lock", "requirements-anpr.lock", "requirements-dev.lock"):
        header = (ROOT / filename).read_text(encoding="utf-8").splitlines()[:3]
        assert "make lock" in "\n".join(header)
