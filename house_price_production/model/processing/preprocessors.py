from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Union

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


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna("Missing")

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )

        return X
    
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X