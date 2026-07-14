import os
import sys

# Taruh folder proyek (e-poliklinik/) di sys.path agar test bisa
# `from storage import ...` tanpa paket.
sys.path.insert(0, os.path.dirname(__file__))
