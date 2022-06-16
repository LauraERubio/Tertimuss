## Installation
A Python version 3.8 is required.

To use Tertimuss, we recommend its installation in a Python virtual environment as a library.

To create and activate a virtual environment (with name .venv) use the following commands:

$ python3 -m venv .venv --copies

# Execute the following line only if you are in a Unix like system (Linux/Mac/FreeBSD)
$ source .venv/bin/activate

# Execute the following line only if you are in a Windows system
$ .\.venv\Scripts\Activate.ps1


To install Tertimuss in the newly created virtual environment use the following command (from the source folder of Tertimuss):


$ pip install .


Also, the library drs must be installed in order to generate task sets and dataclasses-serialization~=1.3.1
You can use the following commands:

$ pip install drs
$ pip install dataclasses-serialization~=1.3.1

## Usage

If you only want to read the results from the given data and solutions execute

$ python3 read_resultsCASE.py -m 2
$ python3 read_resultsCASE.py -m 4

To execute the simulations for the same task sets presented on the article execute script:

simulations.sh

To generate diferent task sets (and overwrite the existing data folder), execute:

$ python3 task_generationCASE.py

NOTE:
This steps are to perform from command line, but if you prefer you can also set up a project using an IDE, we reccomend pyCharm