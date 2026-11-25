import pytest

pytest.importorskip("rich")

from halo import cli  # noqa: E402


def test_parser_commands():
    p = cli.build_parser()
    assert p.parse_args(["ask", "what", "is", "x?"]).question == ["what", "is", "x?"]
    assert p.parse_args(["ingest", "./data", "a.pdf"]).paths == ["./data", "a.pdf"]
    assert p.parse_args(["docs"]).action == "list"
    args = p.parse_args(["docs", "rm", "a.pdf"])
    assert (args.action, args.name) == ("rm", "a.pdf")
    assert p.parse_args(["serve", "--port", "9000"]).port == 9000
    assert p.parse_args([]).command is None  # defaults to chat


def test_docs_list(monkeypatch, capsys):
    import halo.rag.ingest as ingest

    monkeypatch.setattr(ingest, "list_sources", lambda: [{"source": "a.pdf", "chunks": 3, "pages": 2}])
    cli.main(["docs"])
    assert "a.pdf" in capsys.readouterr().out


def test_ingest_missing_path_exits(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["ingest", str(tmp_path / "nope")])
