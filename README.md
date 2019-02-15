

```python
import numpy as np
import pandas as pd

from pairwise_distance import PairwiseDistance
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
a = np.random.randn(1000, 1000)
```


```python
x = PairwiseDistance(batch_size=10, return_self=True).fit(a)
```


```python
# makes sense, because the nearest is always itself
x.knearest(1)[0:5]
```




    array([[0],
           [1],
           [2],
           [3],
           [4]], dtype=int64)




```python
file_url = "https://raw.githubusercontent.com/anaskhan96/Movie-Recommendation/master/trainingset1000.csv"
df = pd.read_csv(file_url, index_col=0)
df.drop_duplicates("Name", inplace=True)
df.reset_index(drop=True, inplace=True)
```


```python
count_v = TfidfVectorizer().fit_transform(df['Name'])
```


```python
movies = PairwiseDistance(batch_size=10, return_self=False, metric="cosine").fit(count_v)
```


```python
kn, dist = movies.knearest(1, return_distance=True)
```


```python
df['Nearest Index'] = kn
df['Distance'] = dist
df['Nearest Movie by Name'] = df['Nearest Index'].map(df['Name'])
```


```python
df[['Name', 'Nearest Movie by Name', 'Distance']].head(20)
```
