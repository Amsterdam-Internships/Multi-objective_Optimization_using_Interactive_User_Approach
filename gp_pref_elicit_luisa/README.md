This repository is contributed by https://github.com/lmzintgraf/gp_pref_elicit. Luisa's papers and code are used to develop further in the problem of Multi-Objective Optimization. Her research involves solving the problem with Gaussian Process with Expected Improvement (non-parametric approach) as an acquisition function whereas I focus on Bayesian Logistic Regression with Thompson Sampling (parametric approach) as an acquisition function. Thus, I first compare Gausiaan Process (GP) with Expected Imporvement (EI) on a synthetic pareto coverage set (PCS) and GP with Thompson Sampling (TS). Then I aim to replace the GP with Bayesian Logistic Regression (BLR)to further compare the two approaches. To run my code, navigate to the ```webInterface``` folder and run the ```BLR-TS.ipynb``` file.

<<<<<<< HEAD
This is the code for the paper

> Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making  
> Luisa M Zintgraf, Diederik M Roijers, Sjoerd Linders, Catholijn M Jonker, Ann Nowe  
> _17th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2018)_

which you can find on [arXiv](https://arxiv.org/abs/1802.07606).

## Main Implementation

The main code, the implementation of pairwise gaussian processes, 
is comprised of `gaussian_process.py`, `acquisition_function.py`, and `dataset.py`.  

For examples on how to use them, we refer to `gp_utilities/utils_experiment.py` and to the `experiments` folder.
There you can also find all experiments we ran for the paper.
If you want to run something and get pretty plots, try
```
python experiments/exp_gp-shape.py
```

## Web Interface

The source code for the user study can be found in the folder `webInterface`.
To run it, type
```
python webInterface/start.py
```
in a terminal and go to `http://0.0.0.0:5000/` in a web browser of your choice 
(I tested a few, but can't guarantee it works everywhere).
There you should find the starting page of the experiment. 
Flags: `-t` to skip the tutorial, `-d` for debug mode.
=======
# Amsterdam Internships - Example README

Explain in short what this repository is. Mind the target audience.
No need to go into too much technical details if you expect some people would just use it as end-users 
and don't care about the internals (so focus on what the code really *does*), not how.
The *_How it works_* section below would contain more technical details for curious people.

---


## Project Folder Structure

Explain briefly what's where so people can find their way around. For example:

There are the following folders in the structure:

1) [`resources`](./resources): Random nice resources, e.g. [`useful links`](./resources/README.md)
1) [`src`](./src): Folder for all source files specific to this project
1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) [`tests`](./tests) Test example
1) [`media`](./media): Folder containing media files (icons, video)
1) ...

OR

Or use something like `tree` to include the overall structure with preferred level of detail (`-L 2` or `-d` or `-a`...)
```buildoutcfg
├── media --> you can still add comments and descriptions in this tree
│   └── examples
├── resources --> a lot of useful links here
├── scripts
├── src --
└── tests
```



If you are lacking ideas on how to structure your code at the first place, take a look at [`CookieCutter`](https://drivendata.github.io/cookiecutter-data-science/)

---


## Installation

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

A person should be able to clone this repo, follow your instructions blindly, and still end up with something *fully working*!

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/InternshipAmsterdamGeneral
    ```

1) If you are using submodules don't forget to include `--recurse-submodules` to the step above or mention that people can still do it afterwards:
   ```bash
   git submodule update --init --recursive
   ```

1) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

Explain example usage, possible arguments, etc. E.g.:

To train... 


```
$ python train.py --some-importang-argument
```

If there are too many command line arguments, you can add a nice table with explanation (thanks, [Diana Epureano](https://www.linkedin.com/in/diana-epureanu-235104153/)!)

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--batch_size`| int| `Batch size.`|  32|
|`--device`| str| `Training device, cpu or cuda:0.`| `cpu`|
|`--early-stopping`|  `store_true`| `Early stopping for training of sparse transformer.`| True|
|`--epochs`| int| `Number of epochs.`| 21|
|`--input_size`|  int| `Input size for model, i.e. the concatenation length of te, se and target.`| 99|
|`--loss`|  str|  `Type of loss to be used during training. Options: RMSE, MAE.`|`RMSE`|
|`--lr`|  float| `Learning rate.`| 1e-3|
|`--train_ratio`|  float| `Percentage of the training set.`| 0.7|
|...|...|...|...|


Alternatively, as a way of documenting the intended usage, you could add a `scripts` folder with a number of scripts for setting up the environment, performing training in different modes or different tasks, evaluation, etc (thanks, [Tom Lotze](https://www.linkedin.com/in/tom-lotze/)!)

---


## How it works

You can explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

>>>>>>> origin/master
