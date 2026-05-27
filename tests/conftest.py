"""Configure test paths for all algorithm modules."""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add each algorithm directory directly to sys.path
for d in [
    '01_astrocyte_tripartite',
    '02_active_dendrite',
    '03_myelin_delay',
    '04_active_inference',
    '05_three_factor_plasticity',
]:
    sys.path.insert(0, os.path.join(ROOT, d))

sys.path.insert(0, ROOT)
