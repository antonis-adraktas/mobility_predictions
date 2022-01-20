# mobility_predictions
This is a python project part of a master thesis that will study bayesian prediction algorythms for a mobility prediction scheme

* In file create_data.py a synthetic data set is created to use as input in the prediction model.  
  
* bayesian_predictor.py is the implementation of the model developed in this project.  

* pybbn_code_appendix.py contains an attempt to build a bayesian network using external library py-bbn. As this library need hypothesis that are not valid in this problem to work without errors it was used only to provide a graphic illustration of the bayesian network.  
  
* shuffle_csv.py is used to created a synthetic data file with balanced samples in the output class based on the originally created files in create_data.py  
  
In the repo the generated data files used in the project are included for direct use.
