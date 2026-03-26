# MQTT AoI Simulation

This project studies **Age of Information (AoI)** minimization in an MQTT-like communication system.  
AoI measures how fresh the received information is — lower AoI means more up-to-date data.

The objective is to simulate and compare different scheduling strategies under limited transmission capacity.

---

## Project Objective

In MQTT systems, a broker cannot transmit all topics at every time slot due to limited capacity.  
This leads to stale updates for some subscribers.

This project aims to:
- Model an MQTT-like topic-subscriber system
- Track AoI evolution over time
- Compare baseline and adaptive scheduling strategies
- Evaluate how scheduling affects information freshness

---

## Implemented Methods

### 1. Round Robin Scheduling
A simple baseline approach that cycles through topics in a fixed order.  
It ensures fairness but does not consider current AoI values.

---

### 2. Price-Based Scheduling (Whittle-inspired)
An adaptive scheduling policy inspired by Whittle index theory.

- Uses AoI values to prioritize updates  
- Applies a lambda-based mechanism for decision making  
- Selects topics dynamically instead of fixed rotation  

---

### 3. Comparative Evaluation
Both methods are evaluated and compared using:
- Average AoI  
- Time evolution plots  
- Performance comparison graphs  

---

## System Model

The simulation parameters are:

- Number of topics (N): 7  
- Number of subscribers (K): 5  
- Time horizon (T): 300  
- Transmission success probability: 0.9  
- Max transmissions per slot: 3  

A fixed subscription matrix is used to ensure fair comparison.

---

## Repository Structure

---

## Main Files

- `src/create_fixed_matrix.py` - generates subscription matrix  
- `src/aoi_simulation.py` - Round Robin simulation  
- `src/price_based_simulation.py` - adaptive scheduling  
- `src/compare_algorithms.py` - comparison plots  
- `src/utils.py` - helper functions  

---

## How to Run

### 1. Generate matrix
```bash
python -m src.create_fixed_matrix