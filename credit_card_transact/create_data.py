# Using numpy (more efficient for large datasets)
import numpy as np
data = np.array([1]*1472 + [0]*1472)
np.random.shuffle(data)

# Using random.sample() for a different approach
import random
data = [1]*1472 + [0]*1472
shuffled_data = random.sample(data, len(data))