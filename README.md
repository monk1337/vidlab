# vidlab
Video frames classification using transformer model


```python
from features import get_features
from model import run_experiment

train_x, train_y, test_x, test_y, label_pr = get_features('all_data')
model = run_experiment(train_x, train_y, test_x, test_y, label_pr, 6)
```
