# MQTT AoI Simulation

This project implements and compares scheduling algorithms for minimizing Age of Information (AoI) in MQTT-based systems.

## Overview

Age of Information (AoI) measures how fresh received data is. This project studies different scheduling strategies under limited transmission capacity.

## Implemented Algorithms

- Round Robin Scheduling
- Price-Based Scheduling (with adaptive lambda update)
- Comparative analysis between both methods

## Project Structure

- `config.py` : simulation parameters  
- `data/` : fixed subscription matrix  
- `results/` : generated plots  
- `src/` : implementation files  
- `old_scripts/` : earlier versions  

## How to Run

```bash
python -m src.create_fixed_matrix
python -m src.aoi_simulation
python -m src.price_based_simulation
python -m src.compare_algorithms