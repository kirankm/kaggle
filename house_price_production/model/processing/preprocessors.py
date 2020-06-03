from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class LabelEncodeCatVar(BaseEstimator, TransformerMixin):
	'''
	Label encode categorical variables
	'''
    def __init__(self, variables:Union[list,str] = None, cat_dict:dict = {}) -> None:
        self.cat_dict = cat_dict
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame) -> "LabelEncodeCatVar":
        for variable in self.variables:
            if variable not in self.cat_dict:
                cat_var = X[variable].astype("category").cat.as_ordered()
                self.cat_dict[variable] = cat_var.cat.categories
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for variable in self.variables:
            cat_var = X[variable].astype("category").cat.as_ordered()
            X[variable] = cat_var.cat.set_categories(self.cat_dict[variable], ordered = True)
        return X

class Numericalize(BaseEstimator, TransformerMixin):
	'''
	convert categorical variables to their numerical codes
	'''
    def __init__(self, variables:Union[str, list] = None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X:pd.DataFrame) -> "Numericalize":
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for variable in self.variables:
            X[variable]  = X[variable].cat.codes + 1
        return X


    