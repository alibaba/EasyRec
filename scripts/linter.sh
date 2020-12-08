#!/usr/bin/env bash
yapf -r -i easy_rec/ pai_jobs/ setup.py
isort -rc easy_rec/ pai_jobs/ setup.py
flake8 easy_rec/ pai_jobs/ setup.py
