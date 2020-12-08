# ========================================================
# makefile
# ========================================================
default: help	# default target


# ========================================================
# Variables
# ========================================================

PYTHON = ""

# https://stackoverflow.com/questions/714100/os-detecting-makefile/12099167#12099167
ifeq ($(OS),Windows_NT)
	PYTHON := python
else
	PYTHON := python3
endif

# --------------------------------------------------------
# Commands

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

notebook: ## Serve locally the jupyter notebook
	$(PIPENV) run jupyter notebook

clean: ## Delete all generated files in project folder
	rm -Rf **/__pycache__
	$(PIPENV) --rm

# --------------------------------------------------------
##@ Commons basics tasks
# --------------------------------------------------------

# source: https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
bash: ## Open a new bash session
	bash

# source: https://suva.sh/posts/well-documented-makefiles/
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<task>\033[0m\n"} /^[0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
