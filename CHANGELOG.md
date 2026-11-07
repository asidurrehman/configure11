# Changelog

Notable changes to CONFIGURE v1.1 are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `config.yaml` holding the run folder and run settings (rainfall and
  configuration numbers, population size, number of generations), plus a
  `--config` option to use another settings file.
- Docstrings for every function in `configure11_functions.py`.
- Detailed explanatory comments throughout the code. These include a
  step-by-step overview and a naming guide at the top of
  `configure11_main.py`, `USER INPUT` markers, and `NOTE (known issue)`
  markers. Comments only; no code changed.
- Test suite (`pytest`) and a GitHub Actions workflow that runs it on Windows
  and Linux with Python 3.10 and 3.12.
- `CITATION.cff`, `pyproject.toml`, `requirements.txt`, `.gitignore` and this
  changelog.

### Changed
- `configure11_main.py` imports `configure11_functions.py` from its own folder
  instead of loading it from `C:\z\configure11\codes`.
- `environment.yaml` is now a short, cross-platform conda environment instead
  of a full export of a Windows Anaconda installation.
- README rewritten with installation, input data, configuration and usage
  instructions.

### Unchanged
- The optimisation, exposure and damage calculations. With the same inputs
  and random seed, results are identical to version 1.1.

## [1.1.0] - 2025-10-15

- Code for Ur Rehman et al. (2025), *Robust Blue-Green Urban Flood Risk
  Management Optimised With a Genetic Algorithm for Multiple Rainstorm Return
  Periods*, Journal of Flood Risk Management, 18(3), e70118.
  <https://doi.org/10.1111/jfr3.70118>
