import os
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "dashboard", "app.py")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "model_store", "meta.json")), reason="run pipeline/train_export.py first")


def run():
    at = AppTest.from_file(APP, default_timeout=120); at.run(); return at


def test_app_loads_and_all_tabs_render():
    at = run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict", "Best candidates", "How accurate is it", "Why", "Stability scenarios", "About the data"]
    assert len(at.dataframe) >= 5 and len(at.metric) >= 6      # tabs produced content (tables + prediction metrics)


def test_prediction_works():
    at = run()
    labels = [m.label for m in at.metric]
    assert "Predicted efficiency (PCE)" in labels and "PCE (direct model)" in labels
    pce = float([m for m in at.metric if m.label == "Predicted efficiency (PCE)"][0].value.split()[0])
    assert 0 < pce < 33
    assert not at.error


def test_invalid_fractions_rejected():
    at = run()
    at.number_input(key="a_Cs").set_value(0.5).run()          # A-site now sums to 1.45
    assert not at.exception
    assert any("Invalid fractions" in e.value for e in at.error)
    assert "Predicted efficiency (PCE)" not in [m.label for m in at.metric]


def test_example_button_and_card():
    at = run()
    at.button(key="btn_example").click().run()
    assert not at.exception and at.session_state["a_FA"] == 0.79
    assert any("Reliability" in m.value for m in at.markdown)
    assert any("Database status" in i.value for i in at.info)


def test_preset_fapbi3():
    at = run()
    at.button(key="pre_FAPbI3").click().run()
    assert not at.exception and at.session_state["a_FA"] == 1.0 and at.session_state["x_I"] == 1.0


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    at = run()
    assert not at.exception and "Predicted efficiency (PCE)" not in [m.label for m in at.metric]
    at.text_input(key="pw_in").set_value("wrong").run(); assert any("Wrong password" in e.value for e in at.error)
    at.text_input(key="pw_in").set_value("secret").run(); assert "Predicted efficiency (PCE)" in [m.label for m in at.metric]
