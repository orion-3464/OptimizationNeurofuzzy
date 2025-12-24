import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dimensions = np.arange(2, 51)  
avg_condition_numbers = []
trials = 1000 

for n in dimensions:
    trial_conds = []
    for _ in range(trials):
        A = np.random.randn(n, n)
        Q = np.dot(A, A.T)
        trial_conds.append(np.linalg.cond(Q))
    
    avg_condition_numbers.append(np.median(trial_conds))

theoretical_trend = (dimensions.astype(float)**2) 

theoretical_trend *= (avg_condition_numbers[0] / theoretical_trend[0]) 

plt.figure(figsize=(10, 6))
plt.plot(dimensions, avg_condition_numbers, marker='o', color='b', label='Observed (Median)')
plt.plot(dimensions, theoretical_trend, linestyle='--', color='r', label='Theoretical Trend ($n^2$)')

plt.title('Condition Number Analysis: $Q = AA^T$')
plt.xlabel('Dimension (n)')
plt.ylabel('Condition Number')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()