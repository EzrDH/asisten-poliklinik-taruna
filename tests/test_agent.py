from types import SimpleNamespace
import agent


def test_chat_menjalankan_tool(monkeypatch):
    panggilan = {"n": 0}

    def fake_chat(model, messages, tools=None):
        panggilan["n"] += 1
        if panggilan["n"] == 1:
            call = SimpleNamespace(
                function=SimpleNamespace(name="rekap_harian", arguments={}))
            return SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[call]))
        return SimpleNamespace(
            message=SimpleNamespace(content="Rekap sudah dibuat.", tool_calls=None))

    dipanggil = {"tool": False}

    def fake_rekap(**kwargs):
        dipanggil["tool"] = True
        return "Rekap: 3 kunjungan."

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    monkeypatch.setitem(agent.T.TOOL_MAP, "rekap_harian", fake_rekap)

    balasan, messages = agent.chat("rekap hari ini")
    assert dipanggil["tool"] is True
    assert "Rekap sudah dibuat." in balasan


def test_chat_tanpa_tool_langsung_balas(monkeypatch):
    def fake_chat(model, messages, tools=None):
        return SimpleNamespace(
            message=SimpleNamespace(content="Halo, ada yang bisa dibantu?",
                                    tool_calls=None))

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    balasan, _ = agent.chat("halo")
    assert "Halo" in balasan
