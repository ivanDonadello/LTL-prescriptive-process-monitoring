# Outcome-Oriented Prescriptive Process Monitoring Based on Temporal Logic Patterns
This repository contains the source code of a new prescriptive process monitoring system that provides process analysts 
with recommendations for achieving a positive outcome of an ongoing process. The recommendations are temporal relations 
between the activities being executed during an ongoing case of a process. The paper describing the system is currently being under review, a preprint can be found 
[here](https://arxiv.org/abs/2211.04880).

## Repository Structure
- `media/input` contains the input logs in `.csv` format. Before reproducing the experiments it is necessary to download 
  and unzip the log folder from [here](https://drive.google.com/file/d/1DDP7OKQhD8cno2tbSpLlIPZ-Mh5y-XUC/view?usp=sharing);
- `media/output` contains the numeric results regarding the performance of the prescriptive system;
- `models` contains the pre-trained models trained on the datasets described in the paper.
- `src` contains the backbone code;
- `settings.py` contains the main settings for the experiments as described in the paper below;
- `dataset_figures.py` is a Python script to extract the dataset figures and save them in a `.csv` file in the 
  `media/output` folder;
- `experiments_runner.py` is the main Python script for running the experiments;
- `gather_results.py` is a Python script for aggregating the results of each dataset and presenting in a more 
  understandable format.
  
## Requirements
The following Python packages are required:

-   [numpy](http://www.numpy.org/) tested with version 1.25.0;
-   [PM4PY](https://pm4py.fit.fraunhofer.de/) tested with version 2.7.4;
-   [sklearn](https://scikit-learn.org/stable/) tested with version 1.2.2;
-   [pandas](https://pandas.pydata.org/) tested with version 2.0.2.
-   [matplotlib](https://matplotlib.org/) tested with version 3.7.1;
-   [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) tested with version 0.10.1;
-   [https://seaborn.pydata.org/](https://seaborn.pydata.org/) tested with version 0.12.2.

## Usage
The system has been tested with Python 3.11.4. After installing the requirements, please download this repository.

### Running the code
To run the evaluation for a given (pretrained) dataset, type:
```
python3 experiments_runner.py --log=Production
```
if you want to train again your model, you need to set the `--load_model` option:
```
$ python experiments_runner.py --log=Production --load_model
```
The DECLARE constraints used for the encoding are grouped into five families: existence, choice, positive relations,
negative relations and all. If you need to use a subset of constraints (e.g., existence and choice) use:
```
$ python experiments_runner.py --log=Production --decl_list="existence,choice"
```
You can also train a model on your own dataset `my_event_log` saved in the standard `.csv` format for event log.
First of all, define you have to add the needed keys to the configuration dictionaries in the `src/dataset_manager/DatasetManager.py` file:
```
dataset = "my_event_log"
filename[dataset] = os.path.join(logs_dir, "my_event_log.csv")
case_id_col[dataset] = "CaseID_my_event_log"
activity_col[dataset] = "ActivityID_my_event_log"
resource_col[dataset] = "ResourceID_my_event_log"
timestamp_col[dataset] = "CompleteTimestamp_my_event_log"
label_col[dataset] = "labelID_my_event_log"
pos_label[dataset] = "pos_label_my_event_log"
neg_label[dataset] = "neg_label_my_event_log"
```
then, you just need to run
```
$ python3 experiments_runner.py --log=my_event_log --load_model=False
```
Type:
```
./run_experiments.sh
```
to run the experiments on the whole pool of datasets in parallel. This works in Unix systems, for Windows systems you
need to create your `.bat` file or run each single dataset as shown above.

### Gathering the results
After running the experiments, type:
```
$ python plot_time_performance.py
```
to aggregate the computational times of generating the recommendations for all datasets. The results will be in the 
file `aggregated_recommendation_times.pdf` in `media/output/result`. 

Type
```
$ python gather_results.py
```
to have an aggregation of the results for each dataset. Such aggregation are found in the files in the `media/output` 
folder.

## Citing
If you use our prescriptive process in your research, please use the following BibTeX entry
```
@article{abs-2211-04880,
  author       = {Ivan Donadello and
                  Chiara Di Francescomarino and
                  Fabrizio Maria Maggi and
                  Francesco Ricci and
                  Aladdin Shikhizada},
  title        = {Outcome-Oriented Prescriptive Process Monitoring Based on Temporal
                  Logic Patterns},
  journal      = {CoRR},
  volume       = {abs/2211.04880},
  year         = {2022}
}
```