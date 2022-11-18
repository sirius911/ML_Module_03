import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_, perf_measure
# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

perf_measure(y, y_hat)
# Accuracy
## your implementation
accuracy_score_(y, y_hat)
## Output:
0.5
## sklearn implementation
accuracy_score(y, y_hat)
## Output:
0.5
# Precision
## your implementation
precision_score_(y, y_hat)
## Output:
0.4
## sklearn implementation
precision_score(y, y_hat)
## Output:
0.4
# Recall
## your implementation
recall_score_(y, y_hat)
## Output:
0.6666666666666666
## sklearn implementation
recall_score(y, y_hat)
## Output:
0.6666666666666666
# F1-score
## your implementation
f1_score_(y, y_hat)
## Output:
0.5
## sklearn implementation
f1_score(y, y_hat)
## Output:
0.5