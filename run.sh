#!/bin/bash

# Change directory to script's directory
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

# Run gunicorn from local virtual environment
./.venv/bin/gunicorn --error-logfile gunicorn-error.log --workers=10 --reload -b localhost:`cat portnum.txt` tileserve:app

