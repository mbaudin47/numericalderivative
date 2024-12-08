# Copyright 2024 - Michaël Baudin.
set -xe
python3 plot_dumontet_vignes_benchmark.py
python3 plot_dumontet_vignes.py
python3 plot_gill_murray_saunders_wright_benchmark.py
python3 plot_gill_murray_saunders_wright.py
python3 plot_numericalderivative.py
python3 plot_openturns.py
python3 plot_stepleman_winarsky_benchmark.py
python3 plot_stepleman_winarsky_plots.py
python3 plot_stepleman_winarsky.py
python3 plot_use_benchmark.py
python3 plot_finite_differences.py

