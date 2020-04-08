from nexus.nexus_inventory.inventory_model.model_boomer import BoomerLogisticRegression, BoomerLinearSVC, BoomerGaussianProcess, BoomerRandomForest, BoomerAdaBoost, BoomerMultinomialNB, BoomerQuadraticDiscriminant
import pytest


@pytest.fixture
def input_data():
    x = [[0],[0],[0],[1],[1]]
    y = [0,0,0,1,1]
    return (x, y)


def test_logistic_regression(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerLogisticRegression(init_args)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert all([a == b for a, b in zip(y, y_pred)])


def test_linear_svc(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerLinearSVC(init_args)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert all([a == b for a, b in zip(y, y_pred)])


def test_gaussian_proc(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerGaussianProcess(init_args)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert all([a == b for a, b in zip(y, y_pred)])


def test_random_forest(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerRandomForest(init_args)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert all([a == b for a, b in zip(y, y_pred)])


def test_ada_boost(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerAdaBoost(init_args)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert all([a == b for a, b in zip(y, y_pred)])


def test_naive_bayes(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerMultinomialNB(init_args)
    model.fit(x, y)
    model.predict(x)


def test_quadratic_discriminant(input_data):
    x, y = input_data
    init_args = {
        'init' : {},
        'call' : {}
    }
    model = BoomerQuadraticDiscriminant(init_args)
    model.fit(x, y)
    model.predict(x)
