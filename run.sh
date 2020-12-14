#!/bin/bash
ipython ./cache_kb/cache.py
python ./evaluation/kb_lookup.py --mode oracle
python ./evaluation/kb_lookup.py --mode top