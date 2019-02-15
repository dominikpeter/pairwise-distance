

```python
import numpy as np
import pandas as pd

from pairwise_distance import PairwiseDistance
from sklearn.feature_extraction.text import TfidfVectorizer
```

    C:\Anaconda3\lib\site-packages\sklearn\utils\fixes.py:313: FutureWarning: numpy not_equal will not check object identity in the future. The comparison did not return the same result as suggested by the identity (`is`)) and will change.
      _nan_object_mask = _nan_object_array != _nan_object_array
    


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Nearest Movie by Name</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>The Raid: Redemption</td>
      <td>0.504737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Godfather</td>
      <td>The Godfather, Part II</td>
      <td>0.368011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Godfather, Part II</td>
      <td>The Godfather Part III</td>
      <td>0.320126</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight</td>
      <td>The Dark Knight Rises</td>
      <td>0.209717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Good, the Bad and the Ugly</td>
      <td>Good Night, and Good Luck.</td>
      <td>0.595508</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pulp Fiction</td>
      <td>Stranger Than Fiction</td>
      <td>0.619053</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12 Angry Men</td>
      <td>X-Men</td>
      <td>0.507281</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Schindler's List</td>
      <td>The Bucket List</td>
      <td>0.539865</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fight Club</td>
      <td>The Breakfast Club</td>
      <td>0.548111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>The Lord of the Rings: The Two Towers</td>
      <td>0.384602</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Inception</td>
      <td>The Shawshank Redemption</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>Star Wars: Episode III - Revenge of the Sith</td>
      <td>0.554783</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Forrest Gump</td>
      <td>The Shawshank Redemption</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>Reign Over Me</td>
      <td>0.742223</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Seven Samurai</td>
      <td>The Last Samurai</td>
      <td>0.479414</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Lord of the Rings: The Two Towers</td>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>0.384602</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Goodfellas</td>
      <td>The Shawshank Redemption</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Star Wars: Episode IV - A New Hope</td>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>0.557934</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Matrix</td>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>0.856459</td>
    </tr>
    <tr>
      <th>19</th>
      <td>City of God</td>
      <td>Aguirre: The Wrath of God</td>
      <td>0.531158</td>
    </tr>
  </tbody>
</table>
</div>


