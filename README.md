# Safety-Monitoring-for-LECPS#

## Project Overview

This project implements and extends ideas from the research paper **“Safety Monitoring for Learning-Enabled Cyber-Physical Systems.”**

The main goal is to monitor the safety of learning-enabled systems even when their operating conditions change. The project uses Signal Temporal Logic (STL) to measure safety and Adaptive Conformal Prediction (ACP) to adapt the monitoring threshold over time.

Along with implementing the core methods, we explored additional scenarios to understand how prediction delays, changing environments, and different adaptation rates affect safety monitoring.

## What We Implemented

### 1. Signal Temporal Logic (STL)

We implemented STL-based safety monitoring to represent safety requirements and calculate robustness values.

These values help us understand not only whether a system violates a safety requirement, but also how close it is to violating one.

### 2. Adaptive Conformal Prediction (ACP)

We implemented ACP to update the monitoring threshold as new observations arrive.

We also evaluated its coverage behaviour under different conditions, including distribution shifts and changing error patterns.

### 3. Testing and Validation

We developed tests to check the correctness of the implementation, validate important properties, and detect unexpected changes in the code.

The tests also help verify that the original implementation remains unchanged where required and that the extended implementation behaves consistently with the baseline in the zero-delay case.

## Our Extensions and New Ideas

### 1. Lag-Aware Adaptive Conformal Prediction

A key extension is **Lag-Aware ACP**, which studies what happens when a prediction is evaluated only after a delay.

In a delayed system, the threshold available when a prediction is finally evaluated may differ from the threshold that was active when the prediction was issued.

Our extension records the threshold associated with each prediction and uses that threshold when evaluating the delayed prediction.

This allows us to compare the original ACP approach with a lag-aware version and investigate how delayed evaluation affects coverage and safety monitoring.

We also check the zero-delay case to verify consistency with the non-delayed setting.

### 2. Extended Synthetic Experiments

We added synthetic scenarios to investigate how safety monitoring behaves under different operating conditions.

These experiments include:

- **Abrupt distribution shifts:** sudden changes in the data.
- **Gradual drift:** data that changes progressively over time.
- **Adversarial monotone changes:** deliberately changing error patterns.
- **Trajectory noise shifts:** changes in the noise affecting trajectories.
- **Reduced LiDAR range:** investigating reduced sensing range, including 5 m, 3 m, and 2 m settings.
- **Changing vehicle counts:** investigating scenarios with 4, 6, and 8 vehicles.

The additional scenarios help us study the behaviour of the monitoring methods under a broader range of conditions instead of relying on a single synthetic setting.

## What We Evaluate

Depending on the experiment, we examine:

- **Coverage:** how often the observed outcomes satisfy the intended coverage target.
- **Threshold behaviour:** how the ACP threshold changes over time.
- **Adaptation behaviour:** how different adaptation rates respond to changing conditions.
- **Effect of delay:** how delayed evaluation affects the monitoring results.
- **Scenario sensitivity:** how the results change with different noise patterns, sensing ranges, and numbers of vehicles.

The results are saved in experiment outputs such as CSV files and plots, making it easier to compare configurations and inspect their behaviour.

## How to Run the Tests

Run the commands below from the project directory.

```bash
python -m pytest test_stl.py
python -m pytest test_acp.py
python -m pytest test_acp_lag_aware.py
```

To run all tests:

```bash
python -m pytest
```

To run an experiment, execute its corresponding Python script. For example:

```bash
python acp_coverage.py
```

```bash
python gamma_comparison.py
```

The exact scripts available depend on the files included in the repository.

## Key Contributions

1. Implemented the core STL-based safety monitoring and ACP methods.
2. Extended ACP to investigate delayed prediction evaluation through a lag-aware approach.
3. Added synthetic scenarios involving distribution shifts, trajectory noise, reduced LiDAR range, and different vehicle counts.
4. Developed tests to validate the implementation and check consistency between baseline and extended behaviour.
5. Generated experimental outputs to support systematic comparisons.

## Conclusion

This project combines implementation, validation, and experimental extension of safety monitoring for learning-enabled systems.

Beyond implementing the core methods, we investigate how delayed evaluation, changes in operating conditions affect monitoring behaviour. These experiments provide a foundation for understanding the strengths and limitations of adaptive safety monitoring in more challenging scenarios.
