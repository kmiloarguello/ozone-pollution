# Ozone Pollution tracker

## Introduction

High concentrations of ground-level ozone affect human health, plants, and animals. Reduce pollution ozone poses a challenge. This research systematically seeks to identify pollution plumes in the Troposphere.

This repository contains the source code used for the internship **Smoothing of incomplete air pollution regions of interest from satellite observations**. The internship was made from March to August 2021.

## Folders

The project was structured to work in a simple python file. 

The `Backup` folder contains information an files used in previous version of the project.
The `Result` folder includes some images of the results.
The `notebooks` folder includes other notebooks used for the project (Those are not necessary in the main file).
The `utils` folder contains some files useful for the main code. As long as new developers want to contribute those folder can be splitted in several files to give modularity to the project.

## Technologies

The project was 100% implemented in `python`. The environment of development were kind of diverse. I used Google Colab and locally I used Linux. For google Colab it has to add some other packages (check `notebooks` folder).

## Project execution

The project can be executed in the main file: `ozone_tracker.ipynb` where there are all the instructions and classes declarations. As long as it is a notebook file, it can be executed by using Anaconda server or google Colab. The execution is by cells, so each one has to be executed one by one.

The source files `(*.nc)` must be added into the project folder for a correct execution, as well as the ground truth images. Normally, those files are included into a folder called `DATA` in the root of the project.

## Code Structure

Inside of the `ozone_tracker.ipynb` file, you'll find several sections that divide the project in Libraries, Class declaration, Independent Functions and Results. The libraries have to be instaled in the project by using Anaconda or Pip. The project mostly were developed in Linux and google Colab. For a file useful for google colab, please check the `notebooks` folder. At the end of the `ozone_tracke.ipynb` you will find some examples plotted by the notebook.

## Troubles

In case of any inconvenience with the code, the execution or the documentation, please open an Issue in this Repo.

## Contributions

This project can be updated by anyone who wants to improve it. There is not a general rule for the contribution, just clone it and try by yourself. I will wait for your PR.

## Author

ARGUELLO Camilo

## Acknowledgment

WENDLING Laurent

KURTZ Camille

VINCENT Nicole

DUFOUR Gaëlle
