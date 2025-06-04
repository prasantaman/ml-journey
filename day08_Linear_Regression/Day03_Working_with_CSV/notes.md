## 📌 Pandas Series:
- A one-dimensional labeled array.
- Each column in a DataFrame is a Series.
- Useful for handling single feature or label.
- Fast vectorized operations possible on Series.

## 📌 Pandas DataFrame:
- A 2D structure made of multiple Series.
- Used to store complete datasets.
- Selecting one column → Series  
  Example: `df['Age'] → Series`
  Example: `df[['Age']] → DataFrame`

## 📌 Series Use in ML & Data Science:
- Input features (X['col']) and target (y) are Series.
- Data cleaning, transformation, visualization often work on Series.
- Output of `model.predict()`, `df['col']`, `df.iloc[:,0]` is often Series.

## 📌 CSV from URL (Online Reading):
1. Use `requests` + `StringIO` + `pandas`:
   - `requests.get()` to fetch file
   - `StringIO(req.text)` to treat string as file
   - `pd.read_csv(StringIO object)` to load as DataFrame

### Example:
```python
import requests
from io import StringIO
import pandas as pd

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
req = requests.get(url)
data = StringIO(req.text)
df = pd.read_csv(data)
