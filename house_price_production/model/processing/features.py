from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


## Transform target variable to log scale
class LogTransformVar(BaseEstimator, TransformerMixin):
	'''
	For log transforming existing variables
	'''
    def __init__(self, variables:Union[list, str] = None, shift:bool = False) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.shift = shift
        
    def fit(self, X:pd.DataFrame) -> "LogTransformVar":
        return self
    
    def transform(self, X:pd.DataFrame)-> pd.DataFrame:
        X = X.copy()
        for variable in self.variables:
            min_value = X[variable].min()
            if min_value > 0:
                pass
            elif self.shift:
                X[variable] = X[variable] + min_value + 1
            else:
                raise ValueError(f'Cannot take logarithm for Variable containing 0 or negative values'
                                f'found non positive value in Varaible: {variable}')
        for variable in self.variables:
            X[variable] = np.log(X[variable])
        return X

