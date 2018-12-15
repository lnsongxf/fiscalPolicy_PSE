SOURCE = src/fiscalPolicy_replicationExercise_OCRCV.ipynb
SCRIPTDIR = bin/
DOCSDIR = docs/
OUTPUTS= plots/fig* docs/fiscalPolicy_replicationExercise_OCRCV.html bin/fiscalPolicy_replicationExercise_OCRCV.py 
.PHONY: clean

all: $(OUTPUTS)

docs/fiscalPolicy_replicationExercise_OCRCV.html: $(SOURCE)
	jupyter nbconvert $^ --to HTML --output-dir $(DOCSDIR)

bin/fiscalPolicy_replicationExercise_OCRCV.py: $(SOURCE)
	jupyter nbconvert --to script $^ --output-dir $(SCRIPTDIR)

plots/fig*: $(SOURCE)
	jupyter nbconvert --execute $^
	rm src/*.html

requirements.txt:
	pipreqs . --force

clean:
	rm -rf $(OUTPUTS)
