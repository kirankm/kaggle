from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from model.processing import preprocessors as pp
from model.config import config


pipeline = Pipeline(
    [
        ("categorical Imputer", pp.CategoricalImputer(variables= config.CATEGORICAL_VARS)),
        ("numerical Imputer", pp.NumericalImputer(variables= config.NUMERICAL_VARS)),
        ("rare_label_encoder Imputer", pp.RareLabelCategoricalEncoder(variables= config.CATEGORICAL_VARS)),
        ("Label_Encoder", pp.LabelEncodeCatVar(variables=config.CATEGORICAL_VARS),
        ("Numericalize", pp.Numericalize(variables= config.CATEGORICAL_VARS))),
        ("DropFeatures", pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES))
    ]
)