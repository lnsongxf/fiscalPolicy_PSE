SOURCE = src/fiscalPolicy_OCRCV.ipynb
SCRIPTDIR = bin/
DOCSDIR = docs/
OUTPUTS= plots/fig* docs/fiscalPolicy_OCRCV.html bin/fiscalPolicy_OCRCV.py 
.PHONY: clean

all: $(OUTPUTS)

docs/fiscalPolicy_OCRCV.html: $(SOURCE)
	jupyter nbconvert --execute --to HTML $^ --output-dir $(DOCSDIR)

bin/fiscalPolicy_OCRCV.py: $(SOURCE)
	jupyter nbconvert --to script $^ --output-dir $(SCRIPTDIR)

plots/fig*: $(SOURCE)
	jupyter nbconvert --execute $^

clean:
	rm -rf $(OUTPUTS)
