[AF]:https://sites.google.com/a/nyu.edu/axelleferriere/home

# README

This assignment is part of the  Fall 2018 Paris Scool of Economics PhD course in Macroeconomics.   

This is to pass [Pr Axelle Ferriere][AF] course. 

## Authors 

- Olimpia Cutinelli Rendina - PhD Student, Paris Scool of Economics
- Cyril Verluise - PhD Student, Paris Scool of Economics

## Programming language and dependecies

- Language: Python 3
- Dependencies: listed in `requirements.txt`
    
## Results

### Get Repo

**Git**

```shell
git --version

cd <your-directory>
git clone https://github.com/cverluise/fiscalPolicy.git 
```

**Download**

- Click `Clone or Download` (green button, upper right)
- Click `Download Zip`

### Read results

1. Make it easy: results are gathered in the `docs/` file. There, you will find:

    - `fiscalPolicy_replicationExercise_OCRCV.html`: html version of the notebook of the replication exercise (code + interactive output). Open it in your favourite browser (Chrome, Safari, etc)
    - `fiscalPolicy_theoreticalExercise_OCRCV.pdf`: pdf of the theoretical exercise 

2. "One look is worth a thousand words": Static plots (pdf format) generated in the course of the replication exercise are available in the `plots/` file. 

3. Go deeper: Interested readers willing to replicate our results should follow the following steps:

```shell
python --version 
pip --version
jupyter --version

cd path/to/fiscalPolicy
pip install -r requirements.txt
jupyter notebook src/fiscalPolicy_replicationExercise_OCRCV.ipynb
```

## Project tree

```
.
├── Makefile
├── README.md
├── bin
│   └── fiscalPolicy_replicationExercise_OCRCV.py
├── docs
│   ├── fiscalPolicy_problemSet.pdf
│   ├── fiscalPolicy_replicationExercise_OCRCV.html
│   └── fiscalPolicy_theoreticalExercise_OCRCV.pdf
├── plots
│   ├── fig1.pdf
│   ├── fig10.pdf
│   ├── fig2.pdf
│   ├── fig3.pdf
│   ├── fig4.pdf
│   ├── fig5.pdf
│   ├── fig6.pdf
│   ├── fig7.pdf
│   ├── fig8.pdf
│   └── fig9.pdf
├── requirements.txt
└── src
    └── fiscalPolicy_replicationExercise_OCRCV.ipynb
```