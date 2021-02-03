# ========================================================
# makefile
# ========================================================
default: help	# default target


# ========================================================
# Variables
# ========================================================

# --------------------------------------------------------
# Commands

PYTHON = python # or python3 if your OS use python2
PIPENV = $(PYTHON) -m pipenv
PYTHON_ENV = $(PIPENV) run $(PYTHON)

# ========================================================
# Targets
# ========================================================

# --------------------------------------------------------
# Utilities tasks (not documented in help)
# --------------------------------------------------------

pipenv: # installs pipenv if necessary
	$(PYTHON) -m pip install pipenv

# --------------------------------------------------------
##@ Project tasks
# --------------------------------------------------------

install: pipenv ## Install dependencies
	$(PIPENV) install
	@# https://stackoverflow.com/questions/62590761/an-error-occured-while-installing-flair-and-pytorch-with-pipenv-in-windows-with
	$(PIPENV) run pip install --no-deps torchvision

clean: ## Delete all generated files in project folder
	rm -Rf **/__pycache__
	rm -Rf src/CBIR/cache
	rm -Rf src/CBIR/data.csv

split-dataset: ## Split coreldb to 3 dataset (test, train, validation) (usage: 'make split-dataset -- --help')
	$(PYTHON_ENV) src/split_dataset.py $(filter-out $@,$(MAKECMDGOALS))

cbir: ## Classify image using CBIR (usage: 'make cbir -- --help')
	cd src/CBIR && $(PYTHON_ENV) scripts/classify.py $(filter-out $@,$(MAKECMDGOALS))

cnn: ## Classify image using CNN (usage: 'make cnn -- --help')
	cd src/CNN && $(PYTHON_ENV) neural_model.py $(filter-out $@,$(MAKECMDGOALS))

# --------------------------------------------------------
##@ Commons basics tasks
# --------------------------------------------------------

# source: https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
bash: ## Open a new bash session
	bash

# source: https://stackoverflow.com/questions/6273608/how-to-pass-argument-to-makefile-from-command-line
%:
	@:

# source: https://suva.sh/posts/well-documented-makefiles/
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<task>\033[0m\n"} /^[0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
