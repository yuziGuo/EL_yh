#!/bin/bash
ipython ./cache_kb/cache.py
ipython ./evaluation/kb_lookup.py --mode oracle
ipython ./evaluation/kb_lookup.py --mode top