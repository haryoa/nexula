from nexula.nexula_inventory.inventory_metrics import nexus_accuracy_score, nexus_f1_score_macro, nexus_f1_score_micro
import pytest


@pytest.fixture
def input_data():
    y_pred = [1,1,0,0,1]
    y_test = [1,1,1,1,1]
    return (y_pred, y_test)


def test_accuracy(input_data):
    y_pred, y_true = input_data
    acc = nexus_accuracy_score(y_true, y_pred)
    assert acc == 0.6


def test_f1_macro_score(input_data):
    y_pred, y_true = input_data
    f1_macro = nexus_f1_score_macro(y_true, y_pred)
    assert f1_macro == pytest.approx(0.37, 0.02)


def test_f1_micro_score(input_data):
    y_pred, y_true = input_data
    f1_macro = nexus_f1_score_micro(y_true, y_pred)
    print(f1_macro)
    assert f1_macro == pytest.approx(0.6, 0.02)