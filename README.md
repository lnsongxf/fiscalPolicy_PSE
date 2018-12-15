# README

Axelle Ferrière's assignment due on December 15th.

PhD course in Macroeconomics (PSE) Fall 2018.   

## Authors 

- Olimpia Cutinelli Rendina, PhD Student - Paris Scool of Economics
- Cyril Verluise, PhD Student - Paris Scool of Economics

## Programming language and dependecies

- Python 3
    
    - requirements.txt
    
## Results

### Get Repo

**Git**
```
cd <your-directory>
git clone https://github.com/cverluise/fiscalPolicy.git 
```

**Download**

- Click `Clone or Download` (green button, upper right)
- Click `Download Zip`

### Read results

Results are gathered in the `docs/` file:

- `fiscalPolicy_replicationExercise_OCRCV.html`: html version of the notebook of the replication exercise (code + interactive output). Open it in your favourite browser (Chrome, Safari, etc)
- `fiscalPolicy_theoreticalExercise_OCRCV.pdf`: pdf of the theoretical exercise 

Static plots (pdf) generated in the course of the replication exercise are also available in the `plots/` file. 

Interested readers willing to replicate our results should follow the following steps:

```shell
python --version 
pip --version
jupyter --version

cd fiscalPolicy
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