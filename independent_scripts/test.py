import numpy as np

# Make a random vector of length 10
vect = np.random.rand(10)
print("Random Vector:", vect)
threshold = 0.5

results = [0.0 if v < threshold else 1.0 for v in vect]
print("Results:", results)
results_flip = [1.0 if v < threshold else 0.0 for v in vect]
print("Results:", results_flip)
results = [0.0 if v > threshold else 1.0 for v in vect]
print("Results:", results)
results_flip = [1.0 if v > threshold else 0.0 for v in vect]
print("Results:", results_flip)