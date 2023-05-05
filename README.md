# esd-2021-95

GitHub repository associated with the research article 'Stable
stadial and interstadial states of the last glacial’s climate
identified in a combined stable water isotope and dust record
from Greenland' presented by Keno Riechers, Leonardo Rydin
Gorjão, Forough Hassanibesheli, Pedro G. Lind, Dirk Witthaut and
Niklas Boers

## Information

In this repository you can find the python scripts used to
generate the figures in esd-2021-95.

The analysis and scripts in this repository are in `python 3`. They are mainly based on standard python libraries as `numpy`, `scipy`, `pandas`, and `matplotlib`. These can be installed easily with, e.g., Anaconda. There are two additional library needed: `kramersmoyal` and `jumpdiff`, which can be installed with

```python
pip install kramersmoyal
pip install jumpdiff
```

## Structure

In the directory [scripts] you can find each script associated with each figure as numbered in the manuscript. The data files are found in [data]. These are

 - `fig01.py` [requires: numpy, matplotlib]
 - `fig01.py` loads `functions.py` [requires: numpy, pandas, scipy]
 - `fig02.py` [requires: numpy, matplotlib]
 - `fig03.py` [requires: numpy, kramersmoyal, matplotlib]
 - `fig04.py` [requires: numpy, kramersmoyal, matplotlib]


The scripts fig02-04 only work after execution of fig01.py,
because fig01 preprocesses the original data stored in [data]. 

## Data

The original data was obtained from the following websites:

all icecore data is available from
https://www.iceandclimate.nbi.ku.dk/data, the links in the
parentheses directly activate the download. 

 - Rasmussen_et_al_2014_QSR_Table_2.xlsx
   [https://www.iceandclimate.nbi.ku.dk/data/Rasmussen_et_al_2014_QSR_Table_2.xlsx,
   last accessed: 04.05.2023]

- NGRIP_dust_on_GICC05_20y_december2014.txt
  [https://www.iceandclimate.nbi.ku.dk/data/NGRIP_dust_on_GICC05_20y_december2014.txt,
  last accessed: 04.05.2023]

- NGRIP_d18O_and_dust_5cm.xls
  [https://www.iceandclimate.nbi.ku.dk/data/NGRIP_d18O_and_dust_5cm.xls,
  last accessed: 04.05.2023]

- GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_Seierstad_et_al._2014_version_10Dec2014-2.xlsx
  [https://www.iceandclimate.nbi.ku.dk/data/GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_Seierstad_et_al._2014_version_10Dec2014-2.xlsx,
  last accessed: 04.05.2023]

the Global average surface temperature shown in Fig. 1 of the
manuscript and used for the detrending is a supplement to

Snyder, C. W.: Evolution of global temperature over
the past two million years, Nature, 538, 226–228,
https://doi.org/10.1038/nature19798, 2016

- 41586_2016_BFnature19798_MOESM245_ESM.xlsx
[https://static-content.springer.com/esm/art%3A10.1038%2Fnature19798/MediaObjects/41586_2016_BFnature19798_MOESM258_ESM.xlsx,
last accessed: 04.05.2023]