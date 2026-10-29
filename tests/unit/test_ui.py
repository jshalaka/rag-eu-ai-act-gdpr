from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.ui_support import shared_service, signed_in_runs


def app_script() -> None:
    from rag_eu_law.app.ui import render  # noqa: PLC0415
    from tests.ui_support import shared_service  # noqa: PLC0415

    render(shared_service())


def _app() -> AppTest:
    return AppTest.from_function(app_script, default_timeout=30).run()


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(b for b in app.button if b.label == label)


def test_page_loads_with_model_and_index_version() -> None:
    app = _app()
    assert not app.exception
    captions = [c.value for c in app.sidebar.caption]
    assert any(c.startswith("Model: ") for c in captions)
    assert any(c.startswith("Index version: ") for c in captions)


def test_ask_button_is_disabled_without_a_question() -> None:
    assert _button(_app(), "Ask").disabled


def test_asking_shows_answer_citation_and_sources_then_saves_a_rating() -> None:
    app = _app()
    app.text_area(key="question").input("Does the data subject have a right to erasure?").run()
    _button(app, "Ask").click().run()
    assert not app.exception
    assert any("[GDPR Art. 3" in m.value for m in app.markdown)
    assert any("Retrieved sources" in e.label for e in app.expander)
    runs = shared_service().recent(10)
    assert runs
    _button(app, "👍").click().run()
    assert shared_service().recent(1)[0].rating == 1
    assert any(s.value == "Rating saved" for s in app.success)


def test_history_tab_lists_the_runs() -> None:
    app = _app()
    app.text_area(key="question").input("What does Article 2 say?").run()
    _button(app, "Ask").click().run()
    assert not app.exception
    assert app.dataframe


def login_script() -> None:
    from rag_eu_law.app.ui import main  # noqa: PLC0415
    from tests.ui_support import shared_container  # noqa: PLC0415

    main(shared_container)


def _login_app() -> AppTest:
    return AppTest.from_function(login_script, default_timeout=30).run()


def test_login_gate_blocks_the_app_until_signed_in() -> None:
    app = _login_app()
    assert [t.value for t in app.title] == ["Sign in"]
    assert not app.text_area


def test_wrong_password_shows_an_error_and_stays_on_login() -> None:
    app = _login_app()
    app.text_input(key="login_email").input("ana@team.test")
    app.text_input(key="login_password").input("nope")
    app.button[0].click().run()
    assert [e.value for e in app.error] == ["Sign-in failed. Check your email and password."]
    assert not app.text_area


def test_signed_in_user_asks_and_the_run_carries_their_email_then_signs_out() -> None:
    app = _login_app()
    app.text_input(key="login_email").input("ana@team.test")
    app.text_input(key="login_password").input("correct horse")
    app.button[0].click().run()
    assert any(c.value == "Signed in as ana@team.test" for c in app.sidebar.caption)
    assert all(t.key != "user" for t in app.sidebar.text_input)
    app.text_area(key="question").input("Does the data subject have a right to erasure?").run()
    _button(app, "Ask").click().run()
    assert not app.exception
    assert signed_in_runs()[0].user == "ana@team.test"
    _button(app, "Sign out").click().run()
    assert [t.value for t in app.title] == ["Sign in"]
