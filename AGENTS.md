The purpose of this project is to familiarize myself with protein fitness optimization workflows.
Specifically, I will be re-implementing the CreiLOV guidance methodology from the ProteinGuide paper: https://doi.org/10.1038/s41587-026-03207-z
CreiLOV mutation datasets can be found in ./data/raw, we will be using the si_002 epistasis dataset

## ProteinGen dependency

- Local source: ../proteingen
- Read ../proteingen/AGENTS.md before working with the library.
- Refer to ../proteingen/docs/ and ../proteingen/examples/.
- Use uv for package management and running Python.

## Modal dependency

- Mostly using a cheap GPU rented from Modal, workflows should support Modal
    
Generally, you can do plenty of scratch work wherever you want but make sure the ending directory is cleanly organized.