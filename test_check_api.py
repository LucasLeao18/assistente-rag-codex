import check_api
import config


def test_sem_chave_retorna_erro(monkeypatch, capsys):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    assert check_api.main() == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out
