Neutron and Hybrid Star Properties in Python
This Python project investigates the properties of neutron and hybrid stars, including:

Mass-radius relations
Stability analysis

Installation
To install the required dependencies, run the following command in your terminal:

Bash
pip install -r requirements.txt
This will install all the necessary Python packages listed in the requirements.txt file.

Usage
The main_op.py and main.py files contain usage examples for one-phase stars (main_op.py) and hybrid stars (main.py). 
You can execute them from the command line with the following syntaxes:

python3 main.py
python3 main_op.py

Options:


Project Structure
The project is organized into the following directories:

equationsofstate: Contains functions to define various equations of state for neutron and hybrid star matter.
star: Implements classes and functions to calculate neutron properties like mass and radius for continuous equations of state.
hybridstar: The same as star, but with emphasis on discontinuous equantios of state, i.e. hybrid stars obtained from Maxwell's construction.
requirements.txt: Lists the required Python dependencies for the project.

Contributing
We welcome contributions to this project! If you'd like to contribute, please feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT license. See the LICENSE file for details.

Disclaimer
The results obtained from this project should be considered for informational purposes only. The accuracy of the calculations depends on the chosen equation of state and may not reflect the true properties of neutron and hybrid stars.
