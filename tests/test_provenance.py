from placepulse_cusp import provenance


def test_git_commit_marks_repository_as_safe(monkeypatch):
    calls = []

    def fake_check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "abc123\n"

    monkeypatch.setattr(provenance.subprocess, "check_output", fake_check_output)

    assert provenance.git_commit() == "abc123"
    command, kwargs = calls[0]
    assert command[:2] == ["git", "-c"]
    assert command[2].startswith("safe.directory=")
    assert "-C" in command
    assert command[-2:] == ["rev-parse", "HEAD"]
    assert kwargs["text"]
