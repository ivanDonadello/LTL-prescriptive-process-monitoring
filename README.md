# Outcome-Oriented Prescriptive Process Monitoring Based on Temporal Logic Patterns
This repo contains the source code of a new prescriptive process monitoring systems that provides users with recommendations 
for achieving a positive outcome of an ongoing process. The recommendations are temporal relations between the activities 
being executed in the process.

## Requirements
The following Python packages are required:

-   [numpy](http://www.numpy.org/) tested with version 1.19.2
-   [PM4PY](https://pm4py.fit.fraunhofer.de/) tested with version 2.2.21
-   [sklearn](https://scikit-learn.org/stable/) tested with version 0.24.1
-   [pandas](https://pandas.pydata.org/) tested with version 1.1.5

## Repository Structure
- `media` contains the input logs and the performance of the prescriptive system;
- `media/input` contains the input logs in `.csv` format. Before reproducing the experiments it is necessary to download 
  and unzip the logs from [here](https://drive.google.com/file/d/1DDP7OKQhD8cno2tbSpLlIPZ-Mh5y-XUC/view?usp=sharing);
- `src` contains the backbone code;
- `settings.py` contains the main settings for the experiments as described in the below paper;
- `dataset_figures.py` is a Python script to extract the dataset figures and save them in a `.csv` file in the 
  `media/output` folder;
- `run_experiments.py` is the main Python script for running the experiments;
- `gather_results.py` is a Python script for aggregating the results of each dataset and presenting in a more 
  understandable format.

## Usage
The system has been tested with Python 3.6.9. After installing the requirements, please download this repository.

### Running the Experiments
Type:
```
 $ python run_experiments.py.py -j num_jobs
```
where `num_jobs` is an integer indicating the number of jobs to execute in parallel. Setting `num_jobs` to -1 will use
all the processors.

### Gathering the Results
Type:
```
 $ python gather_results.py
```
to have an aggregation of the results for each dataset. Such aggregation are found in the files in the `media/output` 
folder.

## Citing
If you use our prescriptive process in your research, please use the following BibTeX entry
```
SOON
```