'''
Environmental variables used throughout the project
'''
import os


def initEnv():

	# root directory of the project
	__DIRROOT__ = os.path.dirname(os.path.realpath(__file__))

	# global parameters
	global __PATHDATA__

	# data path where all data is stored
	__PATHDATA__ = os.path.normpath(os.path.join(__DIRROOT__, '../data'))

initEnv()