# Lost Relatives of the Gumbel Trick

[Matej Balog](http://matejbalog.eu/en/research/), [Nilesh Tripuraneni](https://amplab.cs.berkeley.edu/author/nilesh_tripuraneni/), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Adrian Weller](http://mlg.eng.cam.ac.uk/adrian/)

*34th International Conference on Machine Learning ([ICML 2017](https://2017.icml.cc/))*

[[PDF](http://matejbalog.eu/research/lost_relatives_of_the_gumbel_trick.pdf)]

This repository contains scripts to reproduce experiments appearing in this academic paper.

Requirements:
* Standard Python packages: `argparse`, `json`, `matplotlib`, `numpy`, `scipy`, `sys`
* Only for generating samples yourself for the A\* sampling experiment in [Figure 3a](#figure-3a): [A\* sampling](https://github.com/cmaddis/astar-sampling)
* Only for generating samples yourself for the low-rank perturbation experiments in [Figure 4](#figure-4): [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/) and the `subprocess` Python package

## Instructions

### Figure 1

*Analytically computed MSE and variance of Gumbel and Exponential trick estimators of Z (left) and ln(Z) (right).*
```
python fig1.py
```
![Figure 1](/figures/fig1.png?raw=true "Figure 1")

### Figure 2

*MSE of estimators of Z (left) and ln(Z) (right) stemming from Fréchet (-1/2 < α < 0), Gumbel (α = 0) and Weibull tricks (α > 0).*
```
python fig2.py
```
![Figure 2](/figures/fig2_K100000.png?raw=true "Figure 2")

A faster but less accurate result can be obtained by setting the repetition parameter `K` to a value smaller than the default 100000. For example:
```
python fig2.py --K 1000
```

### Figure 3a
*Sample size M required to reach a given MSE using Gumbel and Exponential trick estimators of ln(Z), using samples from A\* sampling on a Robust Bayesian Regression task.*
```
python fig3a_plot.py
```
![Figure 3a](/figures/fig3a.png?raw=true "Figure 3a")

The plot is produced using 100000 samples stored in `data/astar_rbr_MK100000.json`. To generate samples yourself, please follow these steps:
1. Obtain `astar.py`, `osstar.py`, `heaps.py` and `robustbayesregr.py` from [Chris Maddison's A\* sampling implementation](https://github.com/cmaddis/astar-sampling).
2. Put all these scripts into the same directory where you store this repository.
3. Execute `python fig3a_sample.py`.

### Figure 3b
*MSE of ln(Z) estimators for different values of α, using M=100 samples from the approximate MAP algorithm discussed in Section 5.2, with different error bounds 𝛿.*
```
python fig3b_plot.py
```
![Figure 3b](/figures/fig3b.png?raw=true "Figure 3b")

The plot is produced using sample points stored in `data/bandits_normal_delta0.1_M100000.json`, `data/bandits_normal_delta0.01_M100000.json`, and `data/bandits_normal_delta0.001_M100000.json`.

### Figure 4
*MSEs of U(α) as estimators of ln(Z) on 10x10 attractive (left, middle) and mixed (right) spin glass model with different coupling strengths C.*
```
python fig4_plot.py
```
![Figure 4](/figures/fig4.png?raw=true "Figure 4")

The plot is produced using sample points stored in the `data/` subdirectory. To produce samples yourself, you can follow these steps:
1. Download and compile [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/).
2. Put the file `spin_glass.cpp` into the `examples/` subdirectory of your libDAI installation.
3. Compile `examples/spin_glass.cpp`. For example, execute the following from the libDAI installation directory on Ubuntu:
    ```
    g++ -Iinclude -Wno-deprecated -Wall -W -Wextra -fpic -O3 -g -DDAI_DEBUG  -Llib -oexamples/spin_glass examples/spin_glass.cpp -ldai -lgmpxx -lgmp
    ```
    On Mac the following might work:
    ```
    g++ -Iinclude -I/opt/local/include -Wno-deprecated -Wall -W -Wextra -fPIC -DMACOSX -arch x86_64 -O3 -g -DDAI_DEBUG  -Llib -L/opt/local/lib -o examples/spin_glass examples/spin_glass.cpp -ldai -lgmpxx -lgmp -arch x86_64
    ```
4. Update the `PATH_CPP` variable in `libdai.py` with your libDAI installation location.
5. Execute `python fig4_sample.py`.


## BibTeX
```
@inproceedings{balog2017relatives,
  author = {Matej Balog and Nilesh Tripuraneni and Zoubin Ghahramani and Adrian Weller},
  title={Lost Relatives of the {G}umbel Trick},
  booktitle = {34th International Conference on Machine Learning (ICML)},
  year = {2017},
  month = {August}
}
```
