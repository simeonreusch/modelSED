# modelSED

Allows fitting lightcurves with various model spectra. At the moment only power-law and blackbody fits are implemented.

# Installation
Using Pip: ```pip3 install git+https://github.com/simeonreusch/modelsed```

Otherwise, you can clone the repository: ```git clone https://github.com/simeonreusch/modelsed```

# Usage
```python
import os
from modelSED.sed import SED

# Define the path to the lightcurve you want to model. An example lightcurve is located at https://github.com/simeonreusch/modelsed/data/lightcurves/example.csv
path_to_lightcurve = os.path.join("data", "lightcurves", "example.csv")
```
