from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nexula.nexula_data.data_loader import BaseNexusDataLoader
from nexula.nexula_inventory.inventory_base import NexusBaseModelInventory


class BoomerBase(NexusBaseModelInventory):
    """
    Abstract class for Machine Learning model which uses SKLEARN
    """

    type = 'boomer'

    def __init__(self, model_args, model):
        super().__init__(model_args)
        self.type = 'boomer'
        self.model = model(**self.model_args)

    def fit_to_dataloader(self, data_loader: BaseNexusDataLoader, *args, **kwargs):
        x, y = data_loader.get_all_processed_data(fit_to_data=True)
        self.fit(x, y)

    def fit(self, x_train, y_train, *args, **kwargs):
        self.model.fit(x_train, y_train, **self.fit_args)

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, **kwargs)

    def predict_from_dataloader(self, data_loader, *args, **kwargs):
        x, _ = data_loader.get_all_processed_data(fit_to_data=True)
        return self.predict(x)

    def predict_proba(self, x, *args, **kwargs):
        return self.model.predict_proba(x, **kwargs)

    def predict_proba_from_data_loader(self, data_loader, *args, **kwargs):
        x, _ = data_loader.get_all_processed_data(fit_to_data=True)
        return self.predict_proba(x)

    def prepare_inference(self):
        pass


class BoomerLogisticRegression(BoomerBase):

    def __init__(self, model_args):
        super().__init__(model_args, LogisticRegression)


class BoomerLinearSVC(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, LinearSVC)


class BoomerGaussianProcess(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, GaussianProcessClassifier)


class BoomerRandomForest(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, RandomForestClassifier)


class BoomerAdaBoost(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, AdaBoostClassifier)


class BoomerMultinomialNB(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, MultinomialNB)


class BoomerQuadraticDiscriminant(BoomerBase):
    def __init__(self, model_args):
        super().__init__(model_args, QuadraticDiscriminantAnalysis)
