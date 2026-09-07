from scripts import doctor


def test_doctor_reports_complete_environment(monkeypatch, capsys):
    monkeypatch.setattr(doctor.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(doctor, "version", lambda _name: "1.0")
    assert doctor.main() == 0
    output = capsys.readouterr().out
    assert "Runtime dependency check passed" in output
    assert "[OK] cv2: 1.0" in output


def test_doctor_anpr_mode_reports_missing_external_model(monkeypatch, capsys):
    monkeypatch.setattr(doctor.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(doctor, "version", lambda _name: "1.0")
    assert doctor.main(include_anpr=True) == 1
    output = capsys.readouterr().out
    assert "[MISSING] ANPR plate model" in output
    assert "download_anpr_model.py" in output


def test_doctor_reports_missing_module_and_active_interpreter(monkeypatch, capsys):
    def import_module(name):
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return object()

    monkeypatch.setattr(doctor.importlib, "import_module", import_module)
    monkeypatch.setattr(doctor, "version", lambda _name: "1.0")
    assert doctor.main() == 1
    output = capsys.readouterr().out
    assert "[MISSING] cv2 (opencv-python)" in output
    assert doctor.sys.executable in output
