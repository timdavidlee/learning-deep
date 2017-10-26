
# ML-Lecture2 with Random Forests with in-class Notes

Note, i had to 

    $source activate fastai 

to make sure the library import works otherwise you might have opencv errors.

#### Loading the libraries from the first two blocks with random forests


```python
%load_ext autoreload
%autoreload 2
# load libraries and edit the modules so you can use them


%matplotlib inline
# allows inline plotting
```


```python
import sys
sys.path.append("/Users/tlee010/Desktop/github_repos/fastai/") # go to parent dir

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics
```

### Where does a particular python command it come from?

The left most term is the library where the command comes from 


```python
display
```




    <function IPython.core.display.display>



### what else can we learn about a command?


```python
?display
```

### Source code for the Python?


```python
??display
```

### What parameters does this python function take?

SHIFT + TAB

SHIFT + TAB x2 brings up documentation

SHIFT + TAB x3 new window with documentation


```python
PATH = '/Users/tlee010/Desktop/github_repos/fastai/data/bul'
```

## What is git ignore?

It's a hidden file that keeps track of files that you dont want to be replicated to the server. .gitignore. You can use wildcards to exclude large numbers of files. For instance tmp\* would ignore all the tmp prefixed folders. You can also put it in subdirectories.

This is really useful for ignoring:

- large datasets (don't want to post on github)
- credentials (keys)
- configuration files
- backup files
- scratch files

## What is symlink?

lets you alias folders if you want to point to another folder
    # documentation to create a symlink
    $man lm

## Let's get the data

https://www.kaggle.com/c/bluebook-for-bulldozers/data

Download (need to login) and unzip

If you don't have unzip:

```bash
$brew install unzip
$unzip file.zip
```

#### check the file location


```python
!ls -l /Users/tlee010/kaggle/bulldozers/
```

    total 246368
    -rwxr-xr-x@ 1 tlee010  staff  116403970 Jan 24  2013 [1m[31mTrain.csv[m[m
    -rw-r--r--@ 1 tlee010  staff    9732240 Oct 26 14:20 Train.zip


#### Lets look at the system path

Add { } to force it to use bash PATH instead of python PATH


```python
!ls {PATH}
```

    ls: /Users/tlee010/Desktop/github_repos/fastai/data/bul: No such file or directory


## How can you download straight to AWS? - Firefox trick

CTRL-SHIFT-I for developer console within Firefox

1. If you hit download, and track the actual link that is being used. 
2. Pause it
3. right click the download record, you should get a long curl string It has the cookies / auth details. You can paste in the AWS. Then need to add a '-o filename.zip'. That should download very quickly

### Bulldozers

Predicting the auction sale price for a piece of heavy equipment to create a blue blook for bulldozers.


The key fields are in train.csv are:

- SalesID: the uniue identifier of the sale
- MachineID: the unique identifier of a machine.  A machine can be sold multiple times
- saleprice: what the machine sold for at auction (only provided in train.csv)
- saledate: the date of the sale

### Look at the Data


```python
!head /Users/tlee010/kaggle/bulldozers/Train.csv | head -3
```

    
    
    



```python
!wc  /Users/tlee010/kaggle/bulldozers/Train.csv
```

      401126 8009543 116403970 /Users/tlee010/kaggle/bulldozers/Train.csv


### Using dataframes (Pandas) 


```python
import pandas as pd

df_raw = pd.read_csv('/Users/tlee010/kaggle/bulldozers/Train.csv', low_memory=False, parse_dates=["saledate"])
```

#### Quick note:

We added a `parse_dates` option to the read import to force data typing. For a full list of parameters and options, **check pandas documentation **:

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

### Look at the first few rows


```python
df_raw.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1139246</td>
      <td>66000</td>
      <td>999089</td>
      <td>3157</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>68.0</td>
      <td>Low</td>
      <td>11/16/2006 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1139248</td>
      <td>57000</td>
      <td>117657</td>
      <td>77</td>
      <td>121</td>
      <td>3.0</td>
      <td>1996</td>
      <td>4640.0</td>
      <td>Low</td>
      <td>3/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1139249</td>
      <td>10000</td>
      <td>434808</td>
      <td>7009</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>2838.0</td>
      <td>High</td>
      <td>2/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1139251</td>
      <td>38500</td>
      <td>1026470</td>
      <td>332</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>3486.0</td>
      <td>High</td>
      <td>5/19/2011 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139253</td>
      <td>11000</td>
      <td>1057373</td>
      <td>17311</td>
      <td>121</td>
      <td>3.0</td>
      <td>2007</td>
      <td>722.0</td>
      <td>Medium</td>
      <td>7/23/2009 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1139255</td>
      <td>26500</td>
      <td>1001274</td>
      <td>4605</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>508.0</td>
      <td>Low</td>
      <td>12/18/2008 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1139256</td>
      <td>21000</td>
      <td>772701</td>
      <td>1937</td>
      <td>121</td>
      <td>3.0</td>
      <td>1993</td>
      <td>11540.0</td>
      <td>High</td>
      <td>8/26/2004 0:00</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1139261</td>
      <td>27000</td>
      <td>902002</td>
      <td>3539</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>4883.0</td>
      <td>High</td>
      <td>11/17/2005 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1139272</td>
      <td>21500</td>
      <td>1036251</td>
      <td>36003</td>
      <td>121</td>
      <td>3.0</td>
      <td>2008</td>
      <td>302.0</td>
      <td>Low</td>
      <td>8/27/2009 0:00</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1139275</td>
      <td>65000</td>
      <td>1016474</td>
      <td>3883</td>
      <td>121</td>
      <td>3.0</td>
      <td>1000</td>
      <td>20700.0</td>
      <td>Medium</td>
      <td>8/9/2007 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 53 columns</p>
</div>



### to see all columns ( custom function)


```python
def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
```


```python
display_all(df_raw.tail().transpose())
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>401120</th>
      <th>401121</th>
      <th>401122</th>
      <th>401123</th>
      <th>401124</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SalesID</th>
      <td>6333336</td>
      <td>6333337</td>
      <td>6333338</td>
      <td>6333341</td>
      <td>6333342</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>10500</td>
      <td>11000</td>
      <td>11500</td>
      <td>9000</td>
      <td>7750</td>
    </tr>
    <tr>
      <th>MachineID</th>
      <td>1840702</td>
      <td>1830472</td>
      <td>1887659</td>
      <td>1903570</td>
      <td>1926965</td>
    </tr>
    <tr>
      <th>ModelID</th>
      <td>21439</td>
      <td>21439</td>
      <td>21439</td>
      <td>21435</td>
      <td>21435</td>
    </tr>
    <tr>
      <th>datasource</th>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
    </tr>
    <tr>
      <th>auctioneerID</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>YearMade</th>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>MachineHoursCurrentMeter</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UsageBand</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>saledate</th>
      <td>2011-11-02 00:00:00</td>
      <td>2011-11-02 00:00:00</td>
      <td>2011-11-02 00:00:00</td>
      <td>2011-10-25 00:00:00</td>
      <td>2011-10-25 00:00:00</td>
    </tr>
    <tr>
      <th>fiModelDesc</th>
      <td>35NX2</td>
      <td>35NX2</td>
      <td>35NX2</td>
      <td>30NX</td>
      <td>30NX</td>
    </tr>
    <tr>
      <th>fiBaseModel</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>fiSecondaryDesc</th>
      <td>NX</td>
      <td>NX</td>
      <td>NX</td>
      <td>NX</td>
      <td>NX</td>
    </tr>
    <tr>
      <th>fiModelSeries</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiModelDescriptor</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProductSize</th>
      <td>Mini</td>
      <td>Mini</td>
      <td>Mini</td>
      <td>Mini</td>
      <td>Mini</td>
    </tr>
    <tr>
      <th>fiProductClassDesc</th>
      <td>Hydraulic Excavator, Track - 3.0 to 4.0 Metric...</td>
      <td>Hydraulic Excavator, Track - 3.0 to 4.0 Metric...</td>
      <td>Hydraulic Excavator, Track - 3.0 to 4.0 Metric...</td>
      <td>Hydraulic Excavator, Track - 2.0 to 3.0 Metric...</td>
      <td>Hydraulic Excavator, Track - 2.0 to 3.0 Metric...</td>
    </tr>
    <tr>
      <th>state</th>
      <td>Maryland</td>
      <td>Maryland</td>
      <td>Maryland</td>
      <td>Florida</td>
      <td>Florida</td>
    </tr>
    <tr>
      <th>ProductGroup</th>
      <td>TEX</td>
      <td>TEX</td>
      <td>TEX</td>
      <td>TEX</td>
      <td>TEX</td>
    </tr>
    <tr>
      <th>ProductGroupDesc</th>
      <td>Track Excavators</td>
      <td>Track Excavators</td>
      <td>Track Excavators</td>
      <td>Track Excavators</td>
      <td>Track Excavators</td>
    </tr>
    <tr>
      <th>Drive_System</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Enclosure</th>
      <td>EROPS</td>
      <td>EROPS</td>
      <td>EROPS</td>
      <td>EROPS</td>
      <td>EROPS</td>
    </tr>
    <tr>
      <th>Forks</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pad_Type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ride_Control</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Stick</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Transmission</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Turbocharged</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Extension</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Width</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Enclosure_Type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Engine_Horsepower</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Hydraulics</th>
      <td>Auxiliary</td>
      <td>Standard</td>
      <td>Auxiliary</td>
      <td>Standard</td>
      <td>Standard</td>
    </tr>
    <tr>
      <th>Pushblock</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ripper</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Scarifier</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Tip_Control</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Tire_Size</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Coupler</th>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
    </tr>
    <tr>
      <th>Coupler_System</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grouser_Tracks</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Hydraulics_Flow</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Track_Type</th>
      <td>Steel</td>
      <td>Steel</td>
      <td>Steel</td>
      <td>Steel</td>
      <td>Steel</td>
    </tr>
    <tr>
      <th>Undercarriage_Pad_Width</th>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
    </tr>
    <tr>
      <th>Stick_Length</th>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
    </tr>
    <tr>
      <th>Thumb</th>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
    </tr>
    <tr>
      <th>Pattern_Changer</th>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
    </tr>
    <tr>
      <th>Grouser_Type</th>
      <td>Double</td>
      <td>Double</td>
      <td>Double</td>
      <td>Double</td>
      <td>Double</td>
    </tr>
    <tr>
      <th>Backhoe_Mounting</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Travel_Controls</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Differential_Type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Steering_Controls</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
display_all(df_raw.describe(include='all').transpose())
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>first</th>
      <th>last</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SalesID</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.91971e+06</td>
      <td>909021</td>
      <td>1.13925e+06</td>
      <td>1.41837e+06</td>
      <td>1.63942e+06</td>
      <td>2.24271e+06</td>
      <td>6.33334e+06</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31099.7</td>
      <td>23036.9</td>
      <td>4750</td>
      <td>14500</td>
      <td>24000</td>
      <td>40000</td>
      <td>142000</td>
    </tr>
    <tr>
      <th>MachineID</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.2179e+06</td>
      <td>440992</td>
      <td>0</td>
      <td>1.0887e+06</td>
      <td>1.27949e+06</td>
      <td>1.46807e+06</td>
      <td>2.48633e+06</td>
    </tr>
    <tr>
      <th>ModelID</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6889.7</td>
      <td>6221.78</td>
      <td>28</td>
      <td>3259</td>
      <td>4604</td>
      <td>8724</td>
      <td>37198</td>
    </tr>
    <tr>
      <th>datasource</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>134.666</td>
      <td>8.96224</td>
      <td>121</td>
      <td>132</td>
      <td>132</td>
      <td>136</td>
      <td>172</td>
    </tr>
    <tr>
      <th>auctioneerID</th>
      <td>380989</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.55604</td>
      <td>16.9768</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>99</td>
    </tr>
    <tr>
      <th>YearMade</th>
      <td>401125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1899.16</td>
      <td>291.797</td>
      <td>1000</td>
      <td>1985</td>
      <td>1995</td>
      <td>2000</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>MachineHoursCurrentMeter</th>
      <td>142765</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3457.96</td>
      <td>27590.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3025</td>
      <td>2.4833e+06</td>
    </tr>
    <tr>
      <th>UsageBand</th>
      <td>69639</td>
      <td>3</td>
      <td>Medium</td>
      <td>33985</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>saledate</th>
      <td>401125</td>
      <td>3919</td>
      <td>2009-02-16 00:00:00</td>
      <td>1932</td>
      <td>1989-01-17 00:00:00</td>
      <td>2011-12-30 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiModelDesc</th>
      <td>401125</td>
      <td>4999</td>
      <td>310G</td>
      <td>5039</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiBaseModel</th>
      <td>401125</td>
      <td>1950</td>
      <td>580</td>
      <td>19798</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiSecondaryDesc</th>
      <td>263934</td>
      <td>175</td>
      <td>C</td>
      <td>43235</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiModelSeries</th>
      <td>56908</td>
      <td>122</td>
      <td>II</td>
      <td>13202</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiModelDescriptor</th>
      <td>71919</td>
      <td>139</td>
      <td>L</td>
      <td>15875</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProductSize</th>
      <td>190350</td>
      <td>6</td>
      <td>Medium</td>
      <td>62274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fiProductClassDesc</th>
      <td>401125</td>
      <td>74</td>
      <td>Backhoe Loader - 14.0 to 15.0 Ft Standard Digg...</td>
      <td>56166</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>state</th>
      <td>401125</td>
      <td>53</td>
      <td>Florida</td>
      <td>63944</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProductGroup</th>
      <td>401125</td>
      <td>6</td>
      <td>TEX</td>
      <td>101167</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ProductGroupDesc</th>
      <td>401125</td>
      <td>6</td>
      <td>Track Excavators</td>
      <td>101167</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Drive_System</th>
      <td>104361</td>
      <td>4</td>
      <td>Two Wheel Drive</td>
      <td>46139</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Enclosure</th>
      <td>400800</td>
      <td>6</td>
      <td>OROPS</td>
      <td>173932</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Forks</th>
      <td>192077</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>178300</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pad_Type</th>
      <td>79134</td>
      <td>4</td>
      <td>None or Unspecified</td>
      <td>70614</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ride_Control</th>
      <td>148606</td>
      <td>3</td>
      <td>No</td>
      <td>77685</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Stick</th>
      <td>79134</td>
      <td>2</td>
      <td>Standard</td>
      <td>48829</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Transmission</th>
      <td>183230</td>
      <td>8</td>
      <td>Standard</td>
      <td>140328</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Turbocharged</th>
      <td>79134</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>75211</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Extension</th>
      <td>25219</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>24692</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Width</th>
      <td>25219</td>
      <td>6</td>
      <td>14'</td>
      <td>9615</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Enclosure_Type</th>
      <td>25219</td>
      <td>3</td>
      <td>None or Unspecified</td>
      <td>21923</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Engine_Horsepower</th>
      <td>25219</td>
      <td>2</td>
      <td>No</td>
      <td>23937</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Hydraulics</th>
      <td>320570</td>
      <td>12</td>
      <td>2 Valve</td>
      <td>141404</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pushblock</th>
      <td>25219</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>19463</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ripper</th>
      <td>104137</td>
      <td>4</td>
      <td>None or Unspecified</td>
      <td>83452</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Scarifier</th>
      <td>25230</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>12719</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Tip_Control</th>
      <td>25219</td>
      <td>3</td>
      <td>None or Unspecified</td>
      <td>16207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Tire_Size</th>
      <td>94718</td>
      <td>17</td>
      <td>None or Unspecified</td>
      <td>46339</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Coupler</th>
      <td>213952</td>
      <td>3</td>
      <td>None or Unspecified</td>
      <td>184582</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Coupler_System</th>
      <td>43458</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>40430</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grouser_Tracks</th>
      <td>43362</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>40515</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Hydraulics_Flow</th>
      <td>43362</td>
      <td>3</td>
      <td>Standard</td>
      <td>42784</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Track_Type</th>
      <td>99153</td>
      <td>2</td>
      <td>Steel</td>
      <td>84880</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Undercarriage_Pad_Width</th>
      <td>99872</td>
      <td>19</td>
      <td>None or Unspecified</td>
      <td>79651</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Stick_Length</th>
      <td>99218</td>
      <td>29</td>
      <td>None or Unspecified</td>
      <td>78820</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Thumb</th>
      <td>99288</td>
      <td>3</td>
      <td>None or Unspecified</td>
      <td>83093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pattern_Changer</th>
      <td>99218</td>
      <td>3</td>
      <td>None or Unspecified</td>
      <td>90255</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grouser_Type</th>
      <td>99153</td>
      <td>3</td>
      <td>Double</td>
      <td>84653</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Backhoe_Mounting</th>
      <td>78672</td>
      <td>2</td>
      <td>None or Unspecified</td>
      <td>78652</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Blade_Type</th>
      <td>79833</td>
      <td>10</td>
      <td>PAT</td>
      <td>38612</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Travel_Controls</th>
      <td>79834</td>
      <td>7</td>
      <td>None or Unspecified</td>
      <td>69923</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Differential_Type</th>
      <td>69411</td>
      <td>4</td>
      <td>Standard</td>
      <td>68073</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Steering_Controls</th>
      <td>69369</td>
      <td>5</td>
      <td>Conventional</td>
      <td>68679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


### .. now that EDA is complete, let's model and evaluation

From bulldozer Kaggle overview:

    The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

    Sample submission files can be downloaded from the data page. Submission files should be formatted as follows:

    Have a header: "SalesID,SalePrice"
    Contain two columns
    SalesID: SalesID for the validation set in sorted order
    SalePrice: Your predicted price of the sale
    
#### About metrics:

It's not always MSE, or RMSE, sometimes its profitability, its important to identify and know what the target metric is.


### Let's make it Log sale price


```python
df_raw.SalePrice = np.log(df_raw.SalePrice)
```

#### RandomForestRegressor - continous variable

#### RandomForestClassifier - identifying binary or multiclass categorical variables

RandomForest in general is trivially parallelizable. It's easy to distribute the work load for a cluster of machines. This is triggered by the parameter:

    njobs = -1


```python
#SK LEARN works all teh same way with all algos

# start an ML object (it will start with defaults and empty)
m = RandomForestRegressor(n_jobs=-1)

# then send it the data so the model can be 'FIT'
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-37-c310081c33a1> in <module>()
          1 m = RandomForestRegressor(n_jobs=-1)
    ----> 2 m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
    

    ~/anaconda/envs/fastai/lib/python3.6/site-packages/sklearn/ensemble/forest.py in fit(self, X, y, sample_weight)
        245         """
        246         # Validate or convert input data
    --> 247         X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        248         y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        249         if sample_weight is not None:


    ~/anaconda/envs/fastai/lib/python3.6/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        431                                       force_all_finite)
        432     else:
    --> 433         array = np.array(array, dtype=dtype, order=order, copy=copy)
        434 
        435         if ensure_2d:


    ValueError: could not convert string to float: 'Conventional'


#### Stack Trace - traces the error through all the nested function calls.

### Check the bottom line for the true error ( 'could not convert string to float' - 'conventional')

This dataset contains a mix of **continuous** and **categorical** variables.

The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals.  You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.


```python
add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()
```




    0    2006
    1    2004
    2    2004
    3    2011
    4    2009
    Name: saleYear, dtype: int64



The categorical variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we call `train_cats` to convert strings to pandas categories.

Random forest can handle categorical numbers that are numbers


```python
??add_datepart
```

#### Let's look under the hood

```python
Signature: add_datepart(df, fldname)
Source:   
def add_datepart(df, fldname):
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = (fld - fld.min()).dt.days
    df.drop(fldname, axis=1, inplace=True)
File:      ~/Desktop/github_repos/fastai/fastai/structured.py
Type:      function
```

Let's walk through the function code for this, it uses regex to find the fieldname, and it will replace the date column with multiple columns such as 'day','month','year','etc'. 

#### Are they useful? Include it, and every variant that might be useful. Max, Min, Mean. 

What about the curse of dimensionality - if you have too many columns vs. rows? In practice that doesn't happen. More date the better.

#### No Free Lunch Theorem

For all datasets, there is no one particularly better theorem. (this means all possible datasets, random or otherwise)

#### Free Lunch Theorem

In practice that random forests are the best techique for most cases, mainly because most real world problems are not made of random datasets. 

### Replace categories


```python
??train_cats(df_raw)
```

```python
Signature: train_cats(df)
Source:   
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
File:      ~/Desktop/github_repos/fastai/fastai/structured.py
Type:      function
```


```python
df_raw.UsageBand.cat.categories
```




    Index(['High', 'Low', 'Medium'], dtype='object')



#### What do we do with missing values - lets take a look at the % null


```python
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
```


    Backhoe_Mounting            0.803872
    Blade_Extension             0.937129
    Blade_Type                  0.800977
    Blade_Width                 0.937129
    Coupler                     0.466620
    Coupler_System              0.891660
    Differential_Type           0.826959
    Drive_System                0.739829
    Enclosure                   0.000810
    Enclosure_Type              0.937129
    Engine_Horsepower           0.937129
    Forks                       0.521154
    Grouser_Tracks              0.891899
    Grouser_Type                0.752813
    Hydraulics                  0.200823
    Hydraulics_Flow             0.891899
    MachineHoursCurrentMeter    0.644089
    MachineID                   0.000000
    ModelID                     0.000000
    Pad_Type                    0.802720
    Pattern_Changer             0.752651
    ProductGroup                0.000000
    ProductGroupDesc            0.000000
    ProductSize                 0.525460
    Pushblock                   0.937129
    Ride_Control                0.629527
    Ripper                      0.740388
    SalePrice                   0.000000
    SalesID                     0.000000
    Scarifier                   0.937102
    Steering_Controls           0.827064
    Stick                       0.802720
    Stick_Length                0.752651
    Thumb                       0.752476
    Tip_Control                 0.937129
    Tire_Size                   0.763869
    Track_Type                  0.752813
    Transmission                0.543210
    Travel_Controls             0.800975
    Turbocharged                0.802720
    Undercarriage_Pad_Width     0.751020
    UsageBand                   0.826391
    YearMade                    0.000000
    auctioneerID                0.050199
    datasource                  0.000000
    fiBaseModel                 0.000000
    fiModelDesc                 0.000000
    fiModelDescriptor           0.820707
    fiModelSeries               0.858129
    fiProductClassDesc          0.000000
    fiSecondaryDesc             0.342016
    saleDay                     0.000000
    saleDayofweek               0.000000
    saleDayofyear               0.000000
    saleElapsed                 0.000000
    saleIs_month_end            0.000000
    saleIs_month_start          0.000000
    saleIs_quarter_end          0.000000
    saleIs_quarter_start        0.000000
    saleIs_year_end             0.000000
    saleIs_year_start           0.000000
    saleMonth                   0.000000
    saleWeek                    0.000000
    saleYear                    0.000000
    state                       0.000000
    dtype: float64


## Save the data with feather format


```python
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')
```


```python
??proc_df
```

```python
Signature: proc_df(df, y_fld, skip_flds=None, do_scale=False, preproc_fn=None, max_n_cat=None, subset=None)
Source:   
def proc_df(df, y_fld, skip_flds=None, do_scale=False,
            preproc_fn=None, max_n_cat=None, subset=None):
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    y = df[y_fld].values
    df.drop(skip_flds+[y_fld], axis=1, inplace=True)

    for n,c in df.items(): fix_missing(df, c, n)
    if do_scale: mapper = scale_vars(df)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    res = [pd.get_dummies(df, dummy_na=True), y]
    if not do_scale: return res
    return res + [mapper]
File:      ~/Desktop/github_repos/fastai/fastai/structured.py
Type:      function
```

this will go through each column and run the `fix missing`


```python
??fix_missing
```

```python
Signature: fix_missing(df, col, name)
Source:   
def fix_missing(df, col, name):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum(): df[name+'_na'] = pd.isnull(col)
        df[name] = col.fillna(col.median())
File:      ~/Desktop/github_repos/fastai/fastai/structured.py
Type:      function

```

If it's numeric, create a new column that has the same tells you if missing or not. New column that where missing variables will be replaced with the median value for that continous field.



```python
df, y = proc_df(df_raw, 'SalePrice')
```


```python
df.columns
```




    Index(['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID',
           'YearMade', 'MachineHoursCurrentMeter', 'UsageBand', 'fiModelDesc',
           'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor',
           'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup',
           'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type',
           'Ride_Control', 'Stick', 'Transmission', 'Turbocharged',
           'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower',
           'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control',
           'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks',
           'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width',
           'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type',
           'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
           'Differential_Type', 'Steering_Controls', 'saleYear', 'saleMonth',
           'saleWeek', 'saleDay', 'saleDayofweek', 'saleDayofyear',
           'saleIs_month_end', 'saleIs_month_start', 'saleIs_quarter_end',
           'saleIs_quarter_start', 'saleIs_year_end', 'saleIs_year_start',
           'saleElapsed', 'auctioneerID_na', 'MachineHoursCurrentMeter_na'],
          dtype='object')



## Run your first RandomForest Model


```python
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)
```




    0.98305857537865071



Is this overfitting?

### Setup Test and Train Split


```python
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
```




    ((389125, 66), (389125,), (12000, 66))




```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
```


```python
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
```

    CPU times: user 1min 17s, sys: 666 ms, total: 1min 17s
    Wall time: 15.4 s
    [0.09067208421281527, 0.2523758498048438, 0.98281767127977293, 0.88625205413504937]

