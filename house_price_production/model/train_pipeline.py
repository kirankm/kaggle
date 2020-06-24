import numpy as np 
from sklearn.model_selection import train_test_split
from model.processing.data_management import load_dataset
from model.config import config
from model import pipeline



def run_training() -> None:
	'''
	train the model and save it
	'''
	data = load_dataset(file_name= config.TRAIN_FILE)
	X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], 
                                                   test_size=0.1, random_state=config.RANDOM_STATE)

	y_train = np.log(y_train)
	y_test = np.log(y_test)