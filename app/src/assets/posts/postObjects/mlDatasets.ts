import Post from "../postModel";

export default new Post(
  // title
  "Exploring less common machine learning data sets",
  // subtitle
  "Getting beyond Iris flowers, Titanic survivors, and housing prices",
  // publishDate
  new Date("2020-01-09"),
  // titleImageUrl
  "https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fspecials-images.forbesimg.com%2Fdam%2Fimageserve%2F966248982%2F960x0.jpg%3Ffit%3Dscale",
  // titleImageDescription
  "Finding some new stuff  to learn for our machines.",
  // tags
  ["Data Science & AI/ML", "Learning"],
  // content
  `Probably everyone who starts to learn machine learning will, right away, come across plenty of tutorials or blog posts that uses one of maybe a handful of extremely common data set, such as the [Iris flower data set](https://archive.ics.uci.edu/ml/datasets/iris), the [Titanic survivor data set](https://www.kaggle.com/c/titanic), the [Boston Housing data set](https://www.kaggle.com/c/boston-housing), the [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/), or the [IMDB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/). These data sets are so popular that they're used all over the place, over and over again, and even come delivered with some popular machine learning frameworks such as [Scikit-Learn](https://scikit-learn.org/stable/datasets/index.html) or [Keras](https://keras.io/api/datasets/). They are great for quickly benchmarking or proof-of-concepting a model, but let's face it, they've become pretty boring to everyone who's been emersed in the world of machine learning for a while. There are so many other interesting data sets that are well suited for practicing machine learning problems that have gotten way less attention. Many of which one can find on websites like Kaggle(https://www.kaggle.com/) or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). In this post I want to point out some interesting examples (the list is by no means comprehensive) I used for understanding some machine learning concepts.

I will go through the following five topics/data sets:
- [Regression on Automobile Model Import Data](#automobile)
- [Ensemble methods for Pulsar Neutron Star Classification](#neutronstar)
- [Dimensionality reduction on colonoscopy video data](#colonoscopy)
- [Optimized deep neural networks learning poker hands](#poker)
- [ResNet CNN for classifiying cats and dogs](#catdog)

### Regression on Automobile Model Import Data <a id="automobile"></a>
Let's start with an old data set from 1985 Ward's Automotive Yearbook, which is a great example for a regression problem as it has continuous, ordinal, and categorical (nominal) features which can be used to predict the prices of car models imported into the USA in 1985. You can find the corresponding Jupyter Notebook and data [here](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository/tree/master/Regression-on-automobile-imports/).

The data set was contributed to the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Automobile) by Jeffrey C. Schlimmer, which is where I got it from.

#### Frame the Problem
Let's say we want to have a model that can predict the price of an import model car based on its features (e.g. fuel type, engine type, horse power, mileage etc.). Maybe we are just personally interested to get a car from abroad or we work for a car dealership that is specialized on imports - and of course we want to know what price to expect. Or maybe our model will be only a part in a pipeline of models for an insurance company that needs, among many other inputs, a price prediction to feed to the next model which is supposed to come up with insurance policies for new imports. Let's go with the latter example here.

We clearly have a multivariate regression problem at hand and data set with very few instances (and no new instances streaming in) which means it can easily be batch-processed. To evaluate the performance of our model, we will choose the coefficient of determination, *R<sup>2<sup>*, which is basically the proportion of the variance in our price predictions that is predictable from the feature (the closer it is to 1, the better).


\`\`\`python
# import libraries 
import numpy as np # numerical computation
import pandas as pd # data handling
# visulalization
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("darkgrid")
%matplotlib notebook
\`\`\`

#### Data preparation
Let's import the data into data frame hand have a look.


\`\`\`python
# load data and show first 5 rows
auto = pd.read_csv("imports-85.data",header=None,names=["symboling","normalized_losses","make","fuel_type","aspiration","num_of_doors","body_style","drive_wheels","engine_location","wheel_base","length","width","height","curb_weight","engine_type","num_of_cylinders","engine_size","fuel_system","bore","stroke","compression_ratio","horsepower","peak_rpm","city_mpg","highway_mpg","price"],na_values="?")
auto.head()
\`\`\`




<div class="post-page-table">
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
      <th>symboling</th>
      <th>normalized_losses</th>
      <th>make</th>
      <th>fuel_type</th>
      <th>aspiration</th>
      <th>num_of_doors</th>
      <th>body_style</th>
      <th>drive_wheels</th>
      <th>engine_location</th>
      <th>wheel_base</th>
      <th>...</th>
      <th>engine_size</th>
      <th>fuel_system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression_ratio</th>
      <th>horsepower</th>
      <th>peak_rpm</th>
      <th>city_mpg</th>
      <th>highway_mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>NaN</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>NaN</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>NaN</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164.0</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




\`\`\`python
# print info
auto.info()
\`\`\`

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
    symboling            205 non-null int64
    normalized_losses    164 non-null float64
    make                 205 non-null object
    fuel_type            205 non-null object
    aspiration           205 non-null object
    num_of_doors         203 non-null object
    body_style           205 non-null object
    drive_wheels         205 non-null object
    engine_location      205 non-null object
    wheel_base           205 non-null float64
    length               205 non-null float64
    width                205 non-null float64
    height               205 non-null float64
    curb_weight          205 non-null int64
    engine_type          205 non-null object
    num_of_cylinders     205 non-null object
    engine_size          205 non-null int64
    fuel_system          205 non-null object
    bore                 201 non-null float64
    stroke               201 non-null float64
    compression_ratio    205 non-null float64
    horsepower           203 non-null float64
    peak_rpm             203 non-null float64
    city_mpg             205 non-null int64
    highway_mpg          205 non-null int64
    price                201 non-null float64
    dtypes: float64(11), int64(5), object(10)
    memory usage: 41.7+ KB
    


\`\`\`python
# print describtion 
auto.describe()
\`\`\`




<div class="post-page-table">
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
      <th>symboling</th>
      <th>normalized_losses</th>
      <th>wheel_base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb_weight</th>
      <th>engine_size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression_ratio</th>
      <th>horsepower</th>
      <th>peak_rpm</th>
      <th>city_mpg</th>
      <th>highway_mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>164.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>205.000000</td>
      <td>203.000000</td>
      <td>203.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>201.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.834146</td>
      <td>122.000000</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>3.329751</td>
      <td>3.255423</td>
      <td>10.142537</td>
      <td>104.256158</td>
      <td>5125.369458</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>13207.129353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.245307</td>
      <td>35.442168</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>0.273539</td>
      <td>0.316717</td>
      <td>3.972040</td>
      <td>39.714369</td>
      <td>479.334560</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>7947.066342</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>65.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7775.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>115.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5200.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>150.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>3.590000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>256.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
    </tr>
  </tbody>
</table>
</div>



We got 205 instances, 26 columns (of which 25 are feature and one is the price which we'll try to predict), data types float, int, and object show that we are dealing with continuous, categorical, and in this case also ordinal data. The feature "symboling" is used by actuarians to asses riskyness of an auto: A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. We also see that there are missing values. We'll have to find a strategy to deal with these later.

We can have a look on how the numerical features are distributed by plotting them in histograms:


\`\`\`python
# plot histograms
auto.hist(bins=20)
plt.tight_layout()
\`\`\`


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAACadoJwAAAgAElEQVR4XuydC/znU53/z1wwKMalXGrYLZtbuqhlo1UiKskl435pI5fC7pCUhBixEbtpi41ICk3JUMolowsxlnajcsnaGpeGjBliJs3l/3h+/M/P5/f5fT7fz7l8bt/v93UeDw/8vuf2eZ33+7zfr3Pe55xxy5YtW2aUhIAQEAJCQAgIASEgBISAEBACDSAwTgSkAZTVhBAQAkJACAgBISAEhIAQEAIJAiIgEgQhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgkgEhIASEgBAQAkJACAgBISAEGkNABKQxqNWQEBACQkAICAEhIASEgBAQAiIgNcvAI488YrbbbjuzxRZbmG984xs1t6bqhUA7CHzyk5803/ve98yll15qttxyy3Y6oVaFgBAIQuCOO+4wBx54oNltt93MmWeeGVTHIBc677zzzJe+9CVzxhlnmN13332QP7UT37bhhhuaV73qVebmm2/uRH+q6MRVV11lPvWpT5kjjzzSHHXUUVFVDoo8ioBEiUF5YRGQcoyUo/8RKCIg73rXu8yjjz5q7r///v7/SH2BEBhQBIadgBxwwAFm9uzZ5sc//rF59atfPWaUB8Xh6xfxFQHpPVIh8lglAapKjkRAqkKyoB4RkJoBVvWdQOCJJ54wzz77rFl33XXNiiuuONInEZBODI86IQR6IrBw4ULz2GOPmZe//OXmla985dChVUZA5s2bZ55++ukEGzBSqheBQSQg2Efs5GqrrWZWX331KABFQKLgG57CIiDDM9b60rEIiIBIKoSAEOg6AmUEpOv9H7T+DSIBqXKMRECqRHOA60oTkK985Svm3/7t38yNN95oWFFhq3evvfZKYm/Hjx8/CoXHH3/ckP9nP/uZefLJJ5NVl7e85S3m0EMPNW94wxtG5c22QazqDTfcYObOnWv23Xdf8+lPf3okPzGV3/zmN829995rnn/++WTF+r3vfa/5yEc+YlZeeeUBHgl9WigCrIxeeOGF5uc//7lBLldaaSUzZcoUs/3225sPfehDZtKkSSYbgmVDOvLatLG973//+82DDz5ofvSjH5m//du/HZMVuaaN9dZbz1x//fVm3LhxXp+QJj/I/OWXX27+8Ic/mDXXXNPss88+5pBDDknq/PWvf22++MUvml/+8pfmr3/9q3nb296W6Az9TKf0Ny5atMicf/755r777jMTJ05Mzr1MmzbNvPa1rx3TR+oEP7bA//jHPyarqLvuuqs5/PDDzY477jjUIWousgWgrD7/53/+ZxIigwyyy7bZZpuZf/qnfzJvf/vbx2BuHRjmwQsuuMBcffXVyXzImDLuH/zgB5Myv/jFL8yXv/zlRAYmTJhgtt122yROm1XKdEo7qHfddZf5+te/bh566KFEF7bZZhtzzDHHmLXWWmtUmbSTsMEGG5j/+I//MP/93/9t5s+fn/Rn4403TvI/99xz5uKLL05kHPmkH5tsskmiW8h/NtEusoe88k3M27TNOUPm8fQOxv/8z/8kuP3mN79J7Miqq65q1llnnUTGkT875/cKwVq8eHGiO5zxevjhh5PuIOechcB+0d8irNAP2n/ggQfM8ssvn4zVJz7xCbP22mt76bLN7GLrnnnmGTNz5kxzyy23mP/93/9NvpvvfP3rX28+/OEPm6233nqkbVtfUWds6Ggvh89XNoM+vIVCLrrZa4GpSKbS8+gLL7yQyMdvf/vbZAf9zjvvNKussoqx+ottQNavueaaRNaR7Q984AOJ7K6wwgreqMyZMyfRqbwzudYeIdec9bFpyZIlSX7k97bbbhtlh5AP+k/4HnIwefJk84//+I/mYx/72JhQvl4hUE899ZQ599xzkzMvzAevec1rEll985vfnHuGOC2Pm266aVKWeQlbg5wzH22++eYj32DnrzzA2jzXpBAsbxH2K2AnuDe96U2JcKAA//AP/5D8N8YPRyYr8Aj1QQcdlAg0goih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8ohZaaWVzAYbbGAOO+wws9VWW436yiVLlpiLLroouWL78ccfN+uss05yxfbBBx9sJkyY0FeI9JLFRYsWmfPOO898//vfN/PmzUvk9MADDzR77rlnX32jOhuHwHPPPWe+9rWvJXP9vffem8z3zPNnnnnmmIoHSTfsxw3r3BAnNf1duq4x99EPn7z9gHZVtsbHLvnkrRJDEZAq0cyp6+c//3lilN7whjeYtdde26Asd999t/nBD35g3vnOd5rzzz+/5h70Z/UvvPCC2WWXXcwf//hH8/zzz4uAdGQYjz76aDN79myzww47mE033TQZm6uuuso88MAD5uSTT05ItU2nnHKKufzyy8c8Mkoe8vZLKpPFQw891Nx6661mv/32S8jYLbfcYn784x+b4447LnnPSGk4EICQb7fdduYVr3iFef3rX29mzZpVSEAGRTfSIzuMc8NwSHbxV9Y15j764ZO36+NVpa3xsUs+eavEUASkSjQ96uKV929+85vmhz/8oXnNa17jUXI4sl5wwQXm61//unn/+9+f/Fs7IN0Y97vuuststtlmo3buWD2BLD799NPmtttuSx4Qvf/++5O/7b///ubEE08c6fz06dPNZZddZmbOnGk23HDDbnxUSS96ySJO5uGHH24+9alPjezWUd0RRxyRYMHvq6++el98pzoZhwDOAzrAY7o8BAdBz9sBGSTdSCM2jHNDnMT0f+k6xtxHP3zy9gPaVdkaH7vkk7dqDEVAqkbUsT5CUz7/+c+bb3/72+aNb3yjY6nhyPboo4+anXbayXzmM58xjz32WBLCJgLS7bEnzOTiiy9OVv8JtTr33HOT3b2bbropCTm0ac6cOWb77bdPnPZp06Z1+6OMMWWyeOyxx5obb7wx2RWaNGnSyPcQonbQQQeZ0047TaFYnR/l6jvYi4AMim64ojaoc4Pr9w9jvpgx99EPn7xdH4cqbY2PXfLJWzWGIiBVI1pQ38KFC43951e/+pVhB4SVYpyXtOPSUHc63Qyrx8TSX3HFFQn5EAHp9HAlnTvmmGPM9ddfb+68887kXAjnPO67774kNCmbOCuy8cYbJ+dDup7KZHHHHXc0q666arKQkE7sCrGwsNdeeyW6rjRcCPQiIIOiG64jOqhzg+v3D2O+mDH30Q+fvF0fhyptjY9d8slbNYYiIFUjWlAfh1RxpG3COcEx2WijjRrqQX80w3bgRz/60eTgMnHUFjftgHR3/LhogXCrbbfdNhkv0s4772yWW2655HxINhGWgoN27bXXdvejjEnCp8pk8c1vfrN5+9vfPvLd6Q/aYostzOabb65zXp0e5Xo614uADIJuuKI2qHOD6/cPY77YMffRD5+8XR6Lqm2Nj13yyVs1hiIgVSNaUB+hJ/wzf/785AYhYhdZJdhyyy0b6kH3m2HVmNCrrbfeemTVWASk2+P27LPPJqv83PjDuY5111036TBhVmuuuWayi5VNe++9t3nqqaeS3b+uJldZZCfnfe97n/nCF74w5lMgw05QxwAAIABJREFUJhxKv+SSS7r6mepXTQj0IiD9rhuukA3q3OD6/cOYr4ox99EPn7xdHY86bI2PXfLJWzWGIiBVI+pYH07J2WefnThtr33tax1LDXY24jm5NelHP/rRyMFdEZDujjkTJ1vgXMV44YUXGlb8ber3lSlXWWxz9ai7kqGeDfsOyCDPDZLufASqGnMf2+GTt6vjVoet8bFLPnmrxlAEpGpEHev705/+lKz098thXMfPCs42d+7cZNWcg7u8FWHTN77xDcM/ELZXv/rVow40BzemgtEIcOMPMat33HFHEn5E+FU69XNsro8sthk/Gz2IqqA2BIb5DMggzw21CUyfV1zlmPvYDp+8XYS4LlvjY5d88laNoQhI1Yg61mfvjN9nn30M91gPe/rtb39rdt11154wcLj5l7/85bBD1fr341xx/ztxq+ziETaXTeecc47hSsF+vAXLRxbbvEGkdUFQBwoR6EVA+lk3yoZ80OeGsu8fxt+rHnMf/fDJ28WxqcvW+Ngln7xVYygCUjWimfqIdV9jjTXGtMIVvNwCxHV1HMod9kTsaN6NSbyTQkgWV/LykCO7JErtIbB06VLDhHXdddf1vGKWG7AglEXvgFx99dWdvYDBRxZvvvnmZCeo6B0Qfs/T//ZGUC03gUAvAtLPutELu2GYG5qQnX5qo44x99EPn7xdxLUuW+Njl3zyVo2hCEjViGbq+9jHPpYcPCc+nvcRnnnmGcPr6L/4xS/MW97yFnPppZcm1/Eq5SOgMyDdkowzzjgjCYdDnvfYY48xnSOskMPnpJNOOslceeWVY15C79eraYtkkTAALpaAbHHonJ0hXkKHqPHCrNLwIMAjm8zxOGbIyyabbGLe/e53JwC8613vGiHdg6YbfN8wzw3DI+Gjv7SuMffRD5+8/TJOVdgaH7vkk7dKDEVAqkQzpy5WirmKlFuveCWXq0l5+Zybcw444IBRL0rX3JW+rF4EpFvDhszy6F5RglDbm91YBeZwOlcqE+vKC9Gc7znkkEP6knQXySLv+3zxi180P/jBD5L3a9Zbbz1z4IEHGm77UhouBCAZPCiWl3DWdt999+SnQdMNvmmY54bhkvKXvrauMffRD5+8/TJOVdgaH7vkk7dKDEVAqkRTdQkBISAEhIAQEAJCQAgIASHQEwEREAmIEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAEREMmAEBACQkAICAEhIASEgBAQAo0hIALSGNRqSAgIASEgBISAEBACQkAICAERkAZkYMMNNzSvetWrzM0339xAa8008a53vcs8+uij5v7772+mQbXSVwg88sgjZrvttjNbbLGF+cY3vlFL36ts44477jAHHnig2W233cyZZ55ZS39VaXMIhMhGlXPaIM75zY2eWuoKAiF61JW+qx/dR0AEpIExGkRjVKWxbmAI1ETDCDRhuKpsI4SAVNl+w8Mz8M2FjE2Vc9ogzvkDLzT6wDEIhOiRYBQCrgiIgLgiFZHvoYceMsstt5xZb731ImrpVtE//OEP5q9//at57Wtf262OqTedQKAJw4X8IYcrrriiWXfddaO+WwQkCr7OFQ6RvyrnNBGQzomEOhSAQIgeBTSjIkOKgAjIkA68PlsI1IlAvxkuEZA6paH5utuWPxGQ5sdcLVaPQNt6VP0XqcYuITBwBASF+c///E/z85//3DzxxBPmZS97WRKH/tGPftRstNFGI9hfddVV5lOf+pQ58sgjzQc/+EHzhS98wdx6663m+eefNxtssEHyd7bks2nZsmXm8ssvN9/61rfM73//e7PaaquZHXfc0fzzP/+zOeKII8zs2bPNj3/8Y/PqV796pGieMUo7PJ/85CfNueeem5SbP3+++Zu/+RvzoQ99yOyxxx65suL6jb6C9vTTT5uLL7446cdjjz1mxo8fb9Zcc03zxje+0ey///7mDW94w0iVeeEK9m+92s2eGeH/GS9wo/3Jkyebf/zHfzQf+9jHRmHo+y3K3y4CacP11a9+1XzpS18yP/jBD8yTTz5p1llnHTN16lTzkY98xIwbN25UR+fNm2fIz3kpZHDSpEmJ/B122GHm7//+70fl7WUc2R258MILDXr+xz/+0bzyla80u+66qzn88MMTfc2eX/LVx/POOy/5prykcyTtyh6th8hfrxCs6667zlx00UXmwQcfTGzKO97xDnPssceas88+23zve98zl156qdlyyy3HzPk33nij+drXvma+853vJDK3xhprmPe///2JvVh++eVH8jO/3nnnnWNsB21+/vOfNyussELyO/+26dRTTzXf/OY3zQUXXGDe+c53Jn/+7W9/a6699lpz++23J/rz5z//2ay11lrJnIp94r9t+tWvfpXo4Zvf/GZzxRVX5A4aMo6s019sqE9K69Rxxx2X2Nif/OQniY3FFvO3zTffPKkya1PpF+1hg9LJ2tIf/ehH5vzzzzfXXHONmTt3bqLfH/jABxL9TmNky86ZMycZq1/84hfJzj3tUz9joLNfxaMaokfU9rvf/c585StfMcgAPg1+0tve9rZkfF7zmteMajAtJ+jUv/3bv5mf/exn5k9/+pP5xCc+kfhCjNl3v/vdRI8Yy0WLFiW69Hd/93fJuO+0006j6nzhhRcSmUI+/vd//9fgt+HX7bXXXolflbU7oXKFz4L/gs/0+OOPJ7vxm222mfmnf/on8/a3v31Un5hf+KYiPc47J4zdu+WWW8z3v//95Ftt8rGTLvj66HWVeQeKgPzXf/1X4qgw6TJYCDok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qEwxxkQEJAr6noVDdMOlN2UEJHuugzpjdjFiytK2awiWK/lxwSg0jwhIKHJ+5erSjRACUkQ2lixdVunFCT5Oox+a/rljbIYIiD/eviXK9MNHlkRAfNGPzy8CEo+hUw1liiIC4gRjVKYYYyICEgW9CEjqsHoegRABqU++Qmu+5557zDXXXGNuv/1288gjjyRXuG+wwQbmsMMOM1tttdWoapcsWWIuuugiw7Xvjz/+ePK47dSpU83BBx9sJkyYENqF2uPOfXbZXW+DiyXIPk5jMLCOBWNshgiII8gR2cr8Kh9ZciUgsWecIj7XqWgVMuvUUAWZREAqANGlijJFEQFxQTEuT4xiioDEYd+rdIhuuPRGOyAuKIXlGYYdkKOPPtrMnj3b7LDDDmbTTTc1zz//vLnqqqvMAw88YE4++WSz7777joB3yimnmMsvv9zsvvvuZvPNNzd33313kpc85A1NdemG7Y8vAXG5DU4EZPRoy3aESn95uTL9qIOAxJ5xKv+quBwxfk5cy/6lRUD8MQsqUaYoIiBBsHoVilFMGREvqL0yh+iGSwMiIC4oheUZBgJy1113mc0228wsv/zyIyAtWrTI7LLLLubpp582t912m5k4caK5//77k7/tv//+5sQTTxzJO336dHPZZZeZmTNnmg033DAI6Lp0QwTEbThibIZtQbbDDeuQXGX6URcByRLxtq6JzsOsCpkNGYuQMiIgIagFlClTFB8C0oVbcIog6LLwx/RNRiRA6B2LhOiGS9V2zP761yWGF8XTKe+FcX6POcfh+pK5QrBcRq+7ec4880xz8cUXm1tuuSUJtTr33HPN+eefb2666SYzZcqUkY7PmTPHbL/99ubwww8306ZNC/qgunRDBMRtOGJshgiIG8Yxucr0QwREDxH6yNe4ZcuWLfMp0C95yxTFh4B04RYcEZB+kbzu9zNEN1y+yjoPXKvLpFL2wngsAXF9yVwExGX0upvnmGOOMddff7258847k3MhnPO47777zK233jqm05wV2XjjjZPzISGpLt0QAXEbDREQN5zaylWmHyIgIiA+sikCkkKrKD43NsbWZ0B881YxYfu26Zo/pm/aAXFF2T9fmRHxr/HFEmkC4hK7XgUByb6O7nqzFW275u2C/g9DCFae3D300ENJuNW2225rzjvvvCTLzjvvbJZbbrnkzEc27bbbbmbx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c+bknTfg5Ovbpmzqssmbr3LsysABwq1XeJUd1xPj5fPjjjtO0O7gY2k29dnk9W1Crdb2QBdAON5//31ZuXKlepsGBL5v377SqVOnWrBwfO2lBBc6YCHi3XffVcQDbw6B2AUfQ0WpuJ0Pc8JHH30k2267rQrXRPgmvmHyBwHTed1Gr/zpXbKW2PSVc0gyjH37ykefK4yRqa4W2zesdL+s6giIb8rJ9hABIkAEiAARIAJEgAgQgWpCgASkmkabfSUCRIAIEAEiQASIABEgAiVGgASkxAPA6okAESACRIAIEAEiQASIQDUhQAJSTaPNvhIBIkAEiAARIAJEgAgQgRIjQAJS4gFg9USACBABIkAEiAARIAJEoJoQIAGpptFmX4kAESACRIAIEAEiQASIQIkRIAEp8QCweiJABIgAESACRIAIEAEiUE0IkIBU02izr0SACBABIkAEiAARIAJEoMQIkICUeABYPREgAkSACBABIkAEiAARqCYESECqabTZVyJABIgAESACRIAIEAEiUGIESEBKPACsnggQASJABIgAESACRIAIVBMCJCDVNNrsKxEgAkSACBABIkAEiAARKDECJCAlHgBWTwSIABEgAkSACBABIkAEqgkBEpBqGm32lQgQASJABIgAESACRIAIlBgBEpASDwCrJwJEgAgQASJABIgAESAC1YQACUg1jTb7SgSIABEgAkSACBABIkAESowACUiJB4DVEwEiQASIABEgAkSACBCBakKABKSaRpt9JQJEgAgQASJABIgAESACJUaABKTEA8DqiQARIAJEgAgQASJABIhANSFAAlJNo82+EgEiQASIABEgAkSACBCBEiNAAlLiAWD1RIAIEAEiQASIABEgAkSgmhAgAUkx2m3btpXvfe97MnPmzBSl+PPpxx9/LJ06dZL99ttP7r33Xn8axpZUBAI33XST3HzzzXL11VfLiSeemOvTZZddJn/7299k/Pjxsv/++3vR19NOO03mzJkjTz75pOywww7WbcrXV+uC+AERiEFgypQpcvnll8v5558vF1xwAfEiAkTAAgH6PRZgOc5KApIC0HIjIC+++KL07dtXunfvLtdcc02dnlMRUwgDP41FgAQkFiJmIALWCJCAWEPGD4hADgH6PaUTBhKQFNiTgKQAj59WHQL5CMinn34qX3zxhbRq1UoaN27sBS7cAfFiGNgIAwRIQAxAYhYikAeB//73v/LRRx8p2wMbxFQ8BEhAUmBNApICPH5adQiUU1gSCUjViWfZdpgEpGyHjg0nAlWNAAlIiuEvREDmz58vt99+u4ojX758uTRt2lQOOeQQ+dWvflUnpjzomO22224yatQomTt3roCZ77777nLRRRdJhw4d6rQUv48dO1ZggP7zn//Id7/7XenWrZv0799fjj76aPnkk08E7UDScfZR3dWxw8GtyDvuuEPF6z/00EPy2WefScuWLaVHjx7yy1/+UurVq5cCter6NCmm7777rtx2222CsLkVK1bINttsIwceeKAa25122qkWiMHQuoEDB8of//hH+cc//iGff/65XHLJJXLGGWfkxh/nLDZs2CC33nqrvPXWW9KoUSN17ufSSy+V73znO7J06VK58cYb1bmmlStXCmT84osvrnM246uvvpIHH3xQ5VuwYIGSkc0220zl7927txx//PF1BtomBKuQvOqCw2dGli1bJpBbtGnRokWqb3vttZecc8458pOf/CRS8CZPniwTJkyQDz74QOnoUUcdJb/5zW/kvPPOy+wMCOYDzA04X7J48WK18rbHHnvIL37xCzn44IPrtBN5kP+5555Ter755pvL9ttvLz/+8Y/V2Abl4b333pPRo0fLq6++KkuWLJEttthCmjdvrs51QXcxRwSTzTy1adMmefjhh+XPf/6zwmvVqlWy7bbbyg9+8AM58sgj5dRTT61o5Q7qMnQTejZjxgyB3OGcUM+ePVWIa/369WvhsHr1arnrrrvkscceUyutNTU1suuuu6qx69y5cx3MnnrqKZX3tddeU2O4ceNGadOmjRx33HFy5plnKj0LpkIE5M4775Rrr71WyQjaAFmwSUcccYSyI/PmzVN68sADD8iHH34o3//+92XatGlqftJhvbBTI0eOlGeeeUa+/PJL+eEPfyinn366sknhpG3n448/LmPGjJGpU6eqvuJMZb9+/eTnP/+5+uT555/PzVXA7fDDD1fnXTAfMlUfAkl0ME6G40KwMJfefffdyifTtniXXXaRn/3sZ9K1a9dag5DEBlXfKH7bYxKQFKOfj4DAeMARBEEAoYBxWrhwobz99tvKycFE/qMf/ShXs3bMYMBhTGAkUDYmekz8cDj+8pe/CIReJzgDIDNwYuBkwDmFoXrhhRfUf+O7IAGB4Xj0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CUoBEoFtaBCO999/X1auXKneuQBR6du3r3Tq1KnOp1gZxoSL7zBZ46V1hGj169dPvRJezum0005Tj+bkS+PHj1cPTyHZ4GCTt5zxq4S2m+hMOY0/ZboSpLK8+mBjU2zmxqzy4oZDLDy9++67injgTSs4SMHHdjECuP0RNuCjjz5Sh28RnotwXXwTTlnlLS9JYGuBgA9zcFa6o0c4HwHB4vaf/vQneeihh9SbIbhRFb4lbk0NJx/yFkNiq4aAFANM1kEEiAARIAJEgAgQASJABIhAYQRIQCghRIAIEAEiQASIABEgAkSACBQNARKQokHNiogAESACRIAIEAEiQASIABEgAaEMEAEiQASIABEgAkSACBABIlA0BEhAigY1KyICRIAIEAEiQASIABEgAkSABIQyQASIABEgAkSACBABIkAEiEDRECABKRrUrIgIEAEiQASIABEgAkSACBABEhDKABEgAkSACBABIkAEiAARIAJFQ4AEpGhQsyIiQASIABEgAkSACBABIkAESEAoA0SACBABIkAEiAARIAJEgAgUDQESkKJBzYqIABEgAkSACBABIkAEiAARIAGhDBABIkAEiAARIAJEgAgQASJQNARIQIoGNSsiAkSACBABIkAEiAARIAJEgASEMkAEiAARIAJEgAgQASJABIhA0RAgASka1KyICBABIkAEiAARIAJEgAgQARIQygARIAJEgAgQASJABIgAESACRUOABKRoULMiIkAEiAARIAJEgAgQASJABEhAKANEgAgQASJABIgAESACRIAIFA0BEpCiQc2KiAARIAJEgAgQASJABIgAESABoQwQASJABIgAESACRIAIEAEiUDQESECKBjUrIgJEgAgQASJABIgAESACRIAEhDJABIgAESACRIAIEAEiQASIQNEQIAEpGtTFr+i0006TOXPmyJNPPik77LBD8RvAGomAYwQuu+wy+dvf/ibjx4+X/fff33Hp3xZ30003yc033yxXX321nHjiibkfilV/Zh1jwZkg8PHHH0unTp1kv/32k3vvvTeTOpIWOmXKFLn88svl/PPPlwsuuMC6GJ/7Zt0ZflCVCLRt21a+973vycyZM437n9R/SlKXcaMqLCMJSBEG9MUXX5S+fftK9+7d5ZprrilCjd9UkVSBitZAVkQELBEoFgEgAbEcmCrP7rOTTgJS5cLJ7ksSUhDlP5noeZK6qnWISECKMPIkIEUAmVVUBQKlJiCffvqpfPHFF9KqVStp3LhxVWDOTsYjYOKYxJeSTQ4SkGxwZanlg8B7770nDRs2lDZt2hg3mgTEGKrEGUlAEkNn/iEJiDlWzEkECiFQagLC0SECUQiQgFAuiEBlIUACkv14koCkxBjMevTo0fLqq6/KkiVLZIsttpDmzZurWOBf/vKXMnLkSBWzHpV0TG7QeN12220q9vzxxx9X5fXu3Vv+7//+T32+fv16mThxoirv/fffV3/74Q9/qGLUe/bsKTU1NbWqyReChRXc/v37y8svv6xCw6644gqpV6+e+vbrr79WdTz44IPy73//WzZt2iQ777yzKv+kk07K5UsJGz/3DIE4OZ47d6785je/kRNOOEFuuOGGyNYjzhyrrddee6107dpV5TniiCPkk08+kfnz58t9992nZOujjz6SZs2aySmnnCL9+vVTMvXWW2/Jn/70J6VH//3vf+XAAw9Uco+43WAKEpB169Yp3Zs3b540aNBAnQkZMGCA0omoNHXqVJk8ebJqy4YNG9RqGPpzxhlnyOabb17rE9sQrGA/H3jgAXVG5cMPP5TvfOc76mzAb3/7W9lqq63qNGvhwoVy/fXXy/PPP6/63a5dOznvvPNks802K0nYpmdiWZTmfPXVV7LvvvvK9ttvXydG/JxzzpGnnnoq8mwHZAdzJM7ZrVixIncG5I477lBz+EMPPSSfffaZtGzZUnr06KHsgZ5ngx1btmyZ4BvEpy9atEgaNWoke+21l6Dun/zkJ5EYQIZvv/12Vffy5culadOmcsghh8ivfvWrOuf9XO6AwD798Y9/lBkzZgjajbOFsA2wI/Xr16/V1n/9618yffp0eeGFF1S/vvzyS2Ub0c5zzz1X/Xc4xc1D3/3ud2t9YoNDUYSJlThFwIVubrnllgVDsGATJkyYIB988IHSo6OOOkrZOszDwTO02iZEdTAYXq9DsKAjd955p/zlL39RNnC77bZT9ubXv/61mt+ZREhAUkgBnCYQBCjJnnvuqSbj1atXC5wKTKRwQuBsPfroo/Lss88qh+fHP/5xrsbOnTsL/tEEBGXACcFkDcODCR3CDKIChwmT9tNPPy1QKH0AF5M76jzyyCOVAxc0AlEE5PPPP1dOH4wDDiSibJ3WrFmjjCSIyTbbbCO77767Ku+1116TlStXKkMzbNiwFIjxUx8RMJHjDh06yOGHH67k4B//+IeaqIMJzgUcCxAByLp26LVjfvrpp8ukSZOUYwWSjokdcguH6aCDDpKzzjpLkQ2QBxAKOO/QFzgwcMh00gQEegcyAxlFPjgi7777rnL4YUzgyAfT4MGDFflAuw444ABVpnbe9tlnH7n77rtr1ZOUgKAf0Ps99thDtt12W0Woli5dqhxctCvogMLg9erVSzmQP/jBD2TXXXdVhuqf//ynmleQv9jnxnyUz2K0qU+fPvLSSy/VurADcy7mWSzYwGHAvKjlGs53x44d1ZjBwddzOGQJcyZkETKAhHJhI7DoA4IcTLATv/jFL9RiE+QY8z3IDOZc1A8y36VLl1rfPPbYYzJw4EBlK3bbbTdld2Bz3n77baWXkJsf/ehHuW9cEZC9995b1Ym6oEP4bxBnLARgEQwXNgQT+oq2oi3Qbcg+7A5kHGTvr3/9ay0SYjIPBS+esMWhGHLEOtwjkFY30aJ85zL+8Ic/KJIA/YZMI6wWPhXkFX+DHupLfJ544gm1MAu5wwIa7J1O8OuwyBCsC7YOixeYB5o0aaLmD8wl0GcsOjGRgKSSAe0MwVkBaw4bFjhDWLGJC8HSxgvfw4BhZSu8WgolgbLssssuylkCm0ZCTDpWn7AjAifr1FNPzTUjTEBQz5lnnqlI0aBBgwSKHUxDhw5VTh1Wr4cMGaIcRSQYWxjP119/XcaMGSOHHXZYKtz4sV8ImMrxqFGj1I4DdsxAKIIJcgP5gcxBtnTSBASrnePGjcs5RnC8unXrpnbt4KhDhrETgYRdOBBhGIKrrrpKfv7zn+fK023FH6688ko5+eST1W/YqcPODFaS4ZTB6dIJBuPCCy9Uzg5uKNpxxx3VTyBNZ599tmB3B8ThkksuyX2TlIDAsUIb2rdvn9MdkAwQKugtdnZ0Qn/hwAEzYKoXD7BTc+mll6psJCDF0RUs3txyyy21bj1744031K4vHOh33nmn1s1rWFTCSibmU4xVcA4H2YT8QK6RUA5kADHos2fPzs2rIBjQgQULFqjdPsiBJqggEyAm0AU4Pnq+h/MPBwZEH7sRwR0SLTdweLDqqpMrAoLy4MhBjnXfYEtgc2CHbr31VrULpBNkGwsKwV2LjRs3qnzAJ0xaTOchlJ8Eh+JIEmtxjUBa3dRyG74F65VXXlG78GHSjgUh2DcsaiEFbxE1CbWEjiBB9uHL6RtIIbOwZVjEw+6IzXkU15j6Uh53QFKMBJykZ555Rq1wRYVX6KJtCAgMh145CzYNq8/YGQk7MciDrXvsjmAVFYZRpyABWbt2rTKWIBNYqfrZz35Wq+dYpf3pT3+qnLRHHnmkzhYhVqVBTNAOOKFMlYOAqRxj8sVOG0LysDMRTHAmsII5bdq0WrsPmoCEiQS+xQ4cQg3hRGHVNpgw6WMLPOyAaycFRB07KsGEFVm0b/Hixeo35EHSK2gjRoxQDmUwwchArrFCBcKjt8aTEpDhw4fnVsJ0PXfddZe6/S54DSoICRYtYPxmzZql6g8m3WYSkOLoGZxlEMIg3nrRB8Qbq/nB8QP5hcxiLsScqB0TkMiHH35YzcXBhAUcjHPw+mgQC+wA5gtrvOeeexQBh8yDjCBBhlEGdqKxIx1OKA/lgnSAiCO5JCDABDuWwaQXHw4++GC1yGCSDj30ULUrBNuok+k8lBQHk3Yxj38IpNVN9ChqBwQLTrBX0G3oZzBhFx+LUkhJCUiUrwb7gEWw8PXu/qFenBaRgKTAGbGwWIWCAYKzpEOWwkWaEhCsnkLwwwnEA3Xk+x2rvzhzsmrVKrWiqlenNAHBORQYLGyV33jjjZE7GCAdiHvEigBWsqMSwnDgKEW1MQWM/LTECJjKMZqpnQSEMyEkAwmrtXDcsOV8//331+qNJiAIHWzRokWt3/T2N1aSoT/BhBVnOGZhp0YTEOyyQL7DCQ4bHLeLL75YhRqClICI4PwUdvDCZz3wPQgICDbObiAMEikpAcGWO2L+gwmOJwxcMIRRr1ZjIeC6666r0w84t3BySUCKoxxRseYYM+yOYf7G/IsVS/3GB3YhEGaF37D4pAlI69atFQEIJy3r2KWDXCP9/ve/lz//+c8CgnPcccfV+ebNN99UK6bHH3+8OkuIhP9GvdhJQRhIOGnShLKx64LkioCALAcJg64bO4kIQYFtwKpyMMwQq24ojBQAACAASURBVMlYIIM+wz5hBwQJC3dY9EJ5OpzTZh5KgkNxJIm1uEYgrW6iPVEEBLt10Fssguld8WDbEe6HcMgkBAS7nbA34XO5el6/6KKL1Bmvak8kICkkABMvdh4QS46EkCs4YQhRguOAsxpIpgQkyoHD94hDhPMChw+OX1TSTlRwBVoTEGzXwwHLZ+hQ3tixYyMdoXBdKAsr3UyVg4CpHKPHetUWjhGcfSQduhe1+q8JCBz88AFc7eRH7Y7k2+rWBCQc7qFHQ68aYwsdYU04BAwSk4+84zuQHxgZhODgTBZSUgIS1c8o/cfWPJxRGCEYo3DKtwNUOVLnX08QSoQ4bWCPa5bhgCCcCotMuEQAoXz4HWeXguc/0BMtr1G7efnkCeF/IOZxCXVhFw0JNgC72XEJi0mwTUiuCAjOu+S7UAX9BsEAPrCDSH//+9/ld7/7neBsYb4EcqIvmrCZh5LgEIcZf/cXgTS6iV5FERBEmmCBCmfuog6FIzwSZ5aSEBDMH1h4Cqe0uujvCCVrGQlIMtxyX2H3AatkEDYQETjniO2Fw4PVLayamRKQfK/oagISFXaiG1KIgOA3EBOEBWAFD20LJ+0QwcjoGMZ80BTzMcWUw8PPDREwkWMUBdkGqYCzgcPoIKRw8LGyif/X54Z0tcHbocJNyefkI19SAoJtb2xvJyEgQVKTlIDouOFgXwsRkKiDyfhWEz3ugBgKsINsegUe8oM5EGGFuNkNoVnYHcOuG8KfEMON8EGERYEQF5JX3awoeUKIB3aTEY6kz3hEdWOnnXZSZ5WQtNME56hQ0hecIE9ap0frYiECAqKGA7aagOCg+dFHH62aiN1ILMohvFdfKIHdGVzQEHTukNd0HkqCgwMRYRElQiCNbqLJxSYg+V5dT6uLJYI/s2pJQBxDizMWiNPF6s+xxx6rrixMS0DiQrDQBZAXGMaoECxM8oiJx+FYxO/DiIYNHggKYiJhbGF0maobgSg51ojgilE4VAjrw8oRnLB8N6RlRUDyhWDBeQQJiQrBApEP3qil+6NXulyEYJkSEKwkAzeGYPmjZ8FYczgsWGjBOMHx1md2cA4E8ywWcoKENe5wahQBwcFznPnLt5sXhQzOOOHgNxa99A57HIJpnR7dN5sQLCy+IQxMH9IPtxHnSHAjY5iAhPPlm4eS4BCHE3/3F4E0upmPgGjblEUIFgmImSyRgJjhZJULN1Idc8wx6hYEHEjUty3ku34tznih8kKH0BF3jlCOQofQcRMDrnTEIUHcpIVQFX1WBOXjGkjUgRhmtDkcu2gFADNXBAJhOdad0rKCm55AQCDf+S5PyIqA4DwSDr8GE8IM4ZiAsOM35EEqdAgdNxCBBLg6hG5KQHAFL1aI4dRBf8OvquvwSe6AFE+VcEYOK/m4tQlzJOQaFxPo28lwSQd2tBEXHjz/gRbGzeFRBATvhCD8LuoK23y91tdJ40xJ3C6ILsMVAUF5CAVDSFgwYXELtyaCVOAMChIO5yPkN+rGPFzaom9gjCMgKCtqHkqCQ/EkiTW5RiCNbuYjIFikwrW6UecxcMYK5BkpKKOwfdixjLI/us/5rvzF72l10TWupS6PBCTFCMDJQfgJnPZg0nHo+gCtNk44pI67z8MpznghP4gDCET4GkTEuCPcBNeaxl3Di3KwqgcjgnLQTrz3oZNekcMhSfx3kKAgDwwyttlhiJkqBwFTOQ72WJ+bwN/w5gZ20KJSVgQEdQVvtULoBnYb4fiE26OvTMUheBwC1PoajDkPr9RmHYKF9uPqYb07GnwMFEYRxhGJBKS4eoZLODDP4bICzN/YndAJ50AgSyC6IN/B8xBxc3iUPKEcLErBwYYThJAuHF7VCVfwgpzigKwOiwVxRUgtdvKw2wf9CiaQIxAbnNHSu31pnZ7gFcPoN+yHthu4WhTx+XDMgmeotM7B5mG3SN/yhnzoJ+wVUtC5s5mHkuBQXEliba4RSKqbaEcUKUC4IGQXsoxHcvUDttAhRILg/EdYRqGTIB9YOML5raiFWhIQ85EnATHHqk5Ofe4CYU0QXggjjAkEF5M/QkH0VaBYZcXqKGJXca88VtVgPPRNDPh3vjMgqBix94gXx+0hOOSHR3PgdGFrEociEfMLIxf3ECHK0jcFwZiAhGy99daqbzjciJ0UOEWI5cfvWA3EVjlCEGA84DTpl9lTQMdPPULARo51szH56rj0MPENdi0rAgJjhJVX6JN+iBA37UBuQTIQNhNMesUUehl8iBAhHjjQCl0N7kIUg4DgFW30AwYPcf76IUKEiuHvCGPBOye4DYupOAjot25Qmz7/oWvW50Dw//qMkf4tCQHBt3DEcVsbdu1wNg/OC0Kr/vOf/6hX1nHWKujY4xuEjICgYlUYu96wPbAFKAM7MzhYG7wa3hUB0Q8Roq/QIThj2CGC3QiHEuI37OxAJ9EvOG362l39SGj4DIjtPGSLQ3EkiLVkhUBS3cxHQIK+EHbysbOHhQf4P7jJEP8dfIhQ90tfqQ0/DnM2Fg0g3/q9KhIQcwkgATHHqk5O3OCBw6K4RQHOOSZ+HLTD7Sk4YBi82g0OPHYwwLr1dYT6Xvk446UrxooZnBKsvIHoIMH4YKLHob4wG496CV2Xpe+jxl3xcL70OyaoA+VjFRaECTeY4LwIVo1xkBC7I+HrVFNAyE89QMBGjnVz4XTg6k1MvjhIq2++CXcnKwKCc0yQTTyMiZuncBgeBB4ryVgQiEq4+hakBXINQg/iAnnGalf4bEgxCAjaiHkBt2FhIQHzB5wz3F6EcwZwMkHy8Oo1U3EQeO6553JvbkBe9IOSeqz0g7NhUhA3hxe6cEGfKcHjZDjfATIBpx1yjJBChOqFL3fADgDCndBe2B44S1gswk2KyI9dan3rnCsCAv3CjhCuBIbdwxW7CO0FSQYhC9sf9AtOIxbNsFMP24jQZLxVArnGpS3BHZAk85ANDsWRINaSFQJJdbMQAYGuwSZgBwSyhJ0NLAZjzoWchmUUZeH6aIRAoj1YwIItCe5Uk4CYSwAJiDlWzEkEiMD/EMBDhAhJYYhQNiKBmHoYxkJXZ2dTM0slAkSACBABIpA9AiQg2WPMGohARSGAlXq8KB5+vK+iOlmEziAkBSvnOvZYV4n3JrCTg5AwhLqFV7+L0DRWQQSIABEgAkQgUwRIQDKFl4UTgcpBAOESCL144403VGw3wkNwJS9TMgT0I4k4/4FwTYSz4UwA/sFZLoRs4pAyExEgAkSACBCBSkOABKTSRpT9IQIZIaDj2HFpAa4ixMFufXYooyorulgcIr7xxhsFVz7i0DEukwC2OOyLW7lwJSwTEXCNAB6dxQF3k4S3ocK3IZp8xzxEgAgQgTgESEDiEOLvRIAIEAEiQAQqBAF9OYlJd0ze6TAph3mIABEgAmEEnBOQ4J3h4coQN467+3XC7QF43wLXGy5evFhdfdajRw91gxQfwqOwEgEiQASIABEgAkSACBCBykMgMwKCq8xwHWAwIc4Z4QU6DR06VL1YjGtkcY8yHoDClYG9e/dWL6syEQEiQASIABEgAkSACBABIlBZCGRGQPBYy4ABA/Kihbv48fBQnz59ZNCgQbl8eJ8CD4nhZWX9+mtlQc7eEAEiQASIABEgAkSACBCB6kUgUwKCB7WQwo984W/6VUvcqoNH7nRauHChetU7jsBU75Cx50SACBABIkAEiAARIAJEoHwRyIyANGnSRL1UjITQK7yUeuqpp+aQwjkPvCOAG2DCqWPHjuoFWpwPYSICRIAIEAEiQASIABEgAkSgchBwTkAWLVokV1xxhdrFaNWqlXz66afqkPmbb76prpa89NJLFXq43x733uPMRzjhdeX169cLXltmIgJEgAgQASJABIgAESACRKByEHBOQKKgwW1X2AGZO3eu4JXfNm3aKILSrFkzmTRpUp1PevXqJUuXLpUZM2ZUDtLsCRHwBIFVq9bKhg0bPWmNWTNqaurLVls1lnJse1QP0/RHf2uGHHPZIFBu8pVGjmxw8SlvXJ+pH9mN1pdfrpMtt2xUMfNwdkjZlxwn1/Yl1v3CN90oCgEBDDNnzhScCRk2bJj07Nkz8x0QKooLcU1fRlKl8k1R0iPhTwnLl6+W9evLi4A0aFBfttlmCynHtkeNfJr+6G/9kajKaUm5yVcaOSrXUYvrM/Uju5EFQcdCULnpSXaIuCs5Tq5d1OSbbhSNgOC8B269ws1YOGCe9RkQKorI1ls3FjjyUQkr4CtXrnUh0wXLSKpUvilK5kAVsYJyNB5NmzaRhg1r6uzcFEuOXQ9PUr1AO6gbrkfj2/Ky1g3Xc3IaOcoOxWxLjusz9SM7/EvhV+XTGZO537W+ZYfst/N6lnOQb7pRNALy+OOPywUXXCC4ZhePDY4cOVLGjBkjWd2CVQpFyVI4k5S97bZbyCYRWbRiXa3PWzVtJPVEZNmy1UmKtfomzljkK8w3RbHqtOeZs5zgsup6lCwXU45d9yupXpCAuB6J2uVlrRuu5+Q0cpQtktmVHtdn2o7ssC+FX5Vm7netb9khSwLiBNsVK1ZI06ZNa5W1bt06OeWUU+Sdd95R5zrw4jl2RLp165b3HZCpU6dKu3btErepFIqSuLEZfQjl+2TFOjn0ulm1anjm4sPle00bkYBkhLvvxWbtZGXR/yhZLqYcu+5TnBNVqD46WK5H49vystYN13NyGjnKDsVsS47rM/UjO/xL4Velmftd61t2yJKAOMH2/PPPl7Vr16oXz1u0aCFLliwRkAm87zFw4EA5++yzc/UMHjxYJk+eXOcldJwRwVmRNKkUipKmvVl864PyxRmLfP2mEclCIr4pM2snK4uWpzFCWbQnbZlJ9QL1UjfSop//+6x1w/WcnEaOskMx25Lj+kz9yA7/UvhVaeZ+1/qWHbIkIE6wxZW7IBzvv/++rFy5UvAeyK677ip9+/aVTp061aoDV+2OHTtWXdMLotK8eXMVntWvXz9p0KBBqvaUQlFSNTiDj31QvjhjQQKSwcDHFJm1k5VFj9IYoSzak7bMpHpBApIW+cLfZ60b+ebkeVceI5s3qC8bNyJo9tsUF+eeRo6yRTK70uP6TAKSHfal8KvSzP0++ECmoxEn16blFMrnm24U7QyIC/BsyiiFoti0rxh5fVC+pErlm6IUY7yKVUfWTlYW/UhjhLJoT9oyk+oFCUha5P0kIO+OOFY1LHhez+SMUxo5yhbJ7EqP6zNtR3bYl8KvSjP3++ADmY5GnFyblkMC4gKplGWUQlFSNtn55z4oX1KlohFxLg65AklAssPWtOSkekECYopwsnxZ60a+ORkEBOQjeF7P5IxTGjlKhlDpv4rrM21HdmNUCr+KBMTdePqmG9wBcTe23pVEAuLdkHjRoKydrCw6mcYIZdGetGXGOVHltIqVFgufvs9aN0hA0o92nO745mSl77E/JZCAZDcWcXLtombfdIMExMWoeloGCYinA1PiZmXtZGXRPRKQb1H1zYhkMd6lKjNr3SABST+ycY4a9SM9xvlKIAHJDts4uXZRs2+6QQLiYlQ9LYMExNOBKXGzsnaysugeCQgJSBZyFS4za90gAUk/inGOmm9OVvoe+1NCJROQNA8euhihOLl2WYeLslyUQQLiAkVPyyAB8XRgStysrJ2sLLpHAkICkoVckYAUA1W3dcQ5aiQgbvEOllbJBAQ2Jvxws8lFEK7QjpNrF/X4phskIC5G1dMySEA8HZgSN4sEpMQDEHjLI8lY+GZESo+muxYkGQ+b2rkDYoNWdN44R436kR7jfCX4QkBMr6228YFKvcgVJ9cuRtU33SABcTGqnpZho3xZdSGpUvmmKC7w+fjjj+u8haPLPemkk2TEiBG5ajZs2CDjxo1Tb+QsXrxYWrZsqd7IOeuss6SmpiZVc7J2slI1Ls/HpTYOrvuUVC/QjkrUDdf4Ji0va90gAUk6Mt9+F6c71I/0GPtOQEyvrbbxgUptY+Lk2sWo+qYbJCAuRtXTMmyUL6suJFUq3xTFBT6agOBBzqOPPrpWkTvuuKPsvffeub8NHTpUJk6cKCeeeKJ06NBBXnnlFZkyZYr07t1bhgwZkqo5WTtZqRpXRAJSypjfpHpBApKFdH1bZta6QQKSfvzidKcSbUd61NyU4MsOiOm11TY+EAmIGxmxKYUExAatMstro3xZdS3OWOSrtxKNiCYg/fv3lwEDBuSFfP78+dK1a1fp06ePDBo0KJdv+PDhMmHCBJk2bZq0bds28ZBl7WQlbliBD7MwDqWM+U2qFyQgWUgXCUi2qLotPU53KtF2uEUweWkkIMmxi/syTq7jvjf53TfdIAExGbUyzUMC4tfABQnIueeeqxrXqFGjOo0cNWqUjB49Wp544glp3bp17veFCxdK586dJY7AxPWaBOQbhLIgNXHY69/TGBvfjIhpn8shX9a6wR2Q9FIQpzvUj/QY5yuBBCQ7bOPk2kXNvukGCYiLUfW0DBIQvwZGE5AmTZrImjVrVOMQenX66afLqaeemmssznnMmzdPZs+eXacDHTt2lPbt26vzIUlT1k5W0nYV+i4LspBFmaZ9T2NsfDMipn0uh3xZ6wYJSHopiNMd6kd6jElAvkHgmYsPl+81bSTLlq3ODtT/lRwn1y4a4JtukIC4GFVPyyAB8WtgFi1aJFdccYXaxWjVqpV8+umn6pD5m2++KWeeeaZceumlqsFdunSRhg0bqjMf4dS9e3dZv369TJ8+PXHnsIq1YcPGxN+X4kOc1/h4+To59LpZueq1cUB/kqSttmosn6xwW6ZpO2pq6gvqTzIW+lvTupjPHAESEHOsSpUzzlHzzckqFU5Z1MsdkCxQ/abMOLl2UbNvukEC4mJUPS2DBMTTgQk0C7ddYQdk7ty58thjj0mbNm0UQWnWrJlMmjSpTgd69eolS5culRkzZvjfOcct/GjpmjoEpM12TVLVkkWZqRrEj0uKAAlISeE3qjzOUfPNyTLqVJlkIgHJbqDi5NpFzb7pBgmIi1H1tAwSEE8HJtSsmTNnCs6EDBs2THr27MkdkIhh4w7It6BwByQ7vSYByQ5bVyXHOWq+OVmu+u1DOSQg2Y1CnFy7qNk33SABcTGqnpZBAuLpwISahfMeuPUKN2PhgDnPgNQdtyzOa2RRpqnEpTE2vhkR0z6XQz4SEP9HKU53qB/ZjSEJSHbYxsm1i5p90w0SEBej6mkZJCCeDkyoWY8//rhccMEFgmt28djgyJEjZcyYMbwFK4BTFmQhizJNJS6NsfHNiJj2uRzykYD4P0pxukP9yG4MSUCywzZOrl3U7JtukIC4GFVPyyAB8WtgVqxYIU2bNq3VqHXr1skpp5wi77zzjjrXgRfPsSPSrVu3vO+ATJ06Vdq1a5e4c1k7WYkbVuDDLMhCFmWa9j2NsfHNiJj2uRzyZa0bvAUrvRTE6Q71Iz3G+UogAckO2zi5dlGzb7pBAuJiVD0tgwTEr4E5//zzZe3aterF8xYtWsiSJUsEZALvewwcOFDOPvvsXIMHDx4skydPrvMSOs6I4KxImpS1k5Wmbfm+zYIsZFGmad/TGBvfjIhpn+PyvfHGG/Lggw/KCy+8ILiyGtdV77zzznLOOecIrp/WSV9nHVXeSSedJCNGjIirKu/vWesGCUjiocl9GKc7laof6ZFLXwIJSHoM85UQJ9cuavZNN0hAXIyqp2WQgPg1MLhyF4Tj/fffl5UrVyoHa9ddd5W+fftKp06dajUWV+2OHTtWXdMLotK8eXMVntWvXz9p0KBBqo5l7WSlalyej7MgC1mUadr3NMbGNyNi2ue4fBdeeKHMmTNHjjrqKNltt93UWzm4inrBggUyZMgQ6d27typCExDozNFHH12rWLyrA4KfNGWtGyQgSUfm2+/idKdS9SM9culLIAFJjyEJyLcIkIBkJ09OSsbtP7j1JpzwjsPKlYXfPyABcTIEFVdI1k5WFoBlQRayKNO073FOVKFyKtXBwlXUe+yxh2y22Wa57iNEERc0LF++XJ577jlFvjUBwYUNuLjBZcpaN0hA0o9WnO5Uon6Y7g4CXVztjodqsXi1ePFiFdaLxStcblJTU5NqAEhAUsFX8OM4uXZRs2+6QQLiYlQzLAMGa5OILFqxLldLq6aNpJ5Irdc5o4hK/fr1ZOGytbXeTkAh5fC6p2+KkuEQF73orJ2sLDqUBVnIokzTvqcxNtWmG9dcc43cdddd8tRTTylnKkhAcH01UqNGjUyhL5gva90gAUk/THG6U4n6Ybo7CHSHDh0qEydOrBO+ix1E7CSmSSQgadAr/G2cXLuo2TfdIAFxMaoZlmHqJEURldbbNiYByXBsyrXorJ2sLHAx1QOburMo8//bOxOgraY/jv/atNhCSBRjTCV7jC2MJWuaypBQGDWJwdgZk4qs/1JjL7IlKgYl+5adiuwUWbMlJVIJvf3nd/K8Pe/z3vvc37lnued57vfOGLzPuefc8zm/3znne1Zp+iaNTWiNiDTPacOdf/756pLOWbNmqWWLBQHC/83LtPjhpVd8oedJJ52UNhn1nmvfgAAxKh71cpLvVKN/SGcH586dq2YM+/btS4MHD66FzScsTpgwgaZOnUodOnRIXQgQIKnRJb6YZNeJEQgChOYbECCCQssyiLSTFBVu3tVHqpmTA0ZMr5MFzIBkWaLZp+26k+Uih1I/0EnbRZzS9E0am9AaEWme04T78ssvVYfqoIMOoptvvllF8eOPP9Jll11GXbt2pTZt2tAvv/yilpt8/PHHdNppp9Ell1ySJin1DneweHmrq2eDDZrTDxF1clRdPWf4EdS0cUOqqeE58LUP//+ff66ZES9cSun6u13xSBNvUp7zdFFn6ezg6NGjacyYMc6OcIcASWOxsndM2gRZCmvFuzS863AQIK4JG8Yv7SRBgBiCztHrECBrClvqWy5Mw6SxyYsAWbp0KfGpbwsXLlQjtyw24h5e984zIDxSzLMl7dq1c1FsVuL8btHyeoNCUQKE/6YEV8ny28YRewKtfBgiqTgCpbODri+xhQBxZyImbYL0q0JrOyBApCWXUThpJwkCJKMCqsBkIUAgQEI3W958zp0p3nzLp8HtueeeiZ/80ksvEe8J4WOqWbikeVzPJOjMgESJksLsNX8nP0mzAWkYhP5OUp7zMgMSNTvYvXt3atKkiTo9rvTp1asX8emK06ZNS13EPPO23nrNnM8UFn9glM9IfIPjiPO3Uj+KCxsVLjW8hBeT7NpGuqH5BgSIjVJ1GAcEiEO4OY0aAgQCJGTT//vvv5WQmDFjhlp2xcuvJA9f4MnLtfhkLD4hK83j2jd09oCU62QtXrxMZc/HqGkaji7fScpzaKO8LljEzQ7yssRWrVrRpEmT6iXbp08fWrRokbrwttKe0lnDON9ot0mLelmLmnFkYSEJGxeu0viF+r0QIKGWzH/fBQESeAFV4Oe57mS5QCL1A520XcQpTT+pE1UunmruYPEILZ/4M336dBo5ciR169ZNipSee+45Ovvss4k33PKxo2ke174BAZKmVOq+k+Q71ewfTKLc7CBmQNbMzhQezIDU9R3MgJjXP6IYslirKPowzUDSThKWYGmCzXFw150sF2ilfqCTtos4pekndaLyKEBqamroggsuoKeeeoqGDx9OvXv3jsSwZMkSatmyZZ3fuFN2wgkn0BdffKFGePm43jSPa9+AAElTKhAgBQJJs4PYA9KszvUEcf4WdRBPlu0Bl69JmyD1qtDEOWZApCWXUTipU0CAZFRAFZis606WCyRSP9BJ20Wc0vRNGpvQGhFpnpPCXXvttXTvvfeq/R7HHntsveBdunRRy0vOOussWrFihbrxvHXr1rRgwQKaMmUKzZ8/XwmYgQMHJiUV+7tr34AASV00tS8m+U61+odkdnDUqFE0duxYnIL1n7VAgESLd3MvtBMDBIgdjs5iiXKgqOMZoy4djDuGN+54R8nt6roZTWos4uKr1kZEl5+L8K47WS6+2YVYsB1n1GWgBRalvpXWL4pHylxwzjLOfv360cyZM2M/Yfz48bTXXnupI3dZcHz99df0+++/q7tBOnXqRCeffDIdcsghRllw7RsQIEbFo15O8p1qbDuks4O8D6pnz56x94Cw33Ts2DF1IWSxskQ6uCqd1eDMS8NWwpUFOoUZmm9AgOiUXgZh45yPP6X4eMaoSwfjBEjc8Y6lt6vbyG5SYwEBYoOyXhyuO1l6XyMLbVsscKq24+T4+MaGYr/kdNq0bEalvpXWL6pZgMgswW0o174BAWJefkm+E1onyzzHRNLZQU5ryJAhNHny5Ho3ofPJcHxCnMkDAWJCr/y7SXZtI+XQfAMCxEapOoxDqv7jzpGPuohQcrqKrSyldarQHMUWjxDicd3JcpFH22LBlQCJumQuahQtrV9AgLiwrrVxuvYNCBDz8kvynWpsO6Szg0yXl2rx0dU8U8jLEzfffHN1KMOAAQOocePGRgUAAWKEr+zLSXZtI+XQfAMCxEapOowDAsQh3JxG7bqT5QIrBMhaqqE1Ii7KO6s4XfsGBIh5ySZ11OAf5ozjYoAAccc2ya5tpByab0CA2ChVh3FAgDiEm9OoXXeyXGCFAIEAcWFXpXG69g0IEPNSTOqohdbJMs9xODFAgLgriyS7tpFyaL4BAWKjVB3GAQHiEG5Oo3bdyXKBNQ8CJG4Te9wGdhec8x6na9+AADG3sKSOWmidLPMchxMDBIi7skiyaxsph+YbECA2StVhHBAgDuHmNGrXnSwXWPMgQKI2sZfbwO6Cc97jdO0bpgIk6gRDvlzsn39W0ZIly3NRfEkdtdA6WdVUKBAg7kozya5tpByab0CA2ChVh3FAgDiEm9OoXXeyXGDNiwAp3cRebgO7C855j9O1b5gKEJ8nGIZqC0kdtdA6WaFyTPNdECBpqMneSbJrWSzlQ4XmGxAgIWHR5wAAGN9JREFUNkrVYRwQIA7h5jRq150sF1ghQJbVYg2tEXFR3lnF6do3bAiQ0pMNfd5VkFW5FKeb1FGDf7grJQgQd2yT7NpGyqH5BgSIjVJ1GAcEiEO4OY3adSfLBVYIEAgQF3ZVGqdr34AAMS/FpI5aaJ0s8xyHEwMEiLuySLJrGymH5hsQIDZK1WEcECAO4eY0atedLOlmah38ECAQIDr2kjasa9+AAElbMmvfS+qohdbJMs9xODFAgESXRVybx6FLDxGJK80ku7ZhBaH5BgSIjVJ1GIcvARK1uVHHeWw7VWiO4rCIvUfto5NVeiN41GZqnYxDgECA6NhL2rA+fCPqskrpRbI+L5FNy9D1e0kdNbQd7koAAiSabdQBIhxSp91LsmsbpRqab0CA2ChVh3H4EiCuNjemdarQHMVhEXuPOotOluk6dQgQCBAfjpKFb3C+bAsQGyOyPninSSOpTUHbkYaq7J28CRDpwGzczKZOu5dk17ISKh8qNN+AALFRqg7j8ClAXGxuTOtUoTmKwyL2HnUWnSydijgKCAQIBIgPR8nCN1wIEBsjsj54p0kjqU1B25GGquydvAkQ6cAsBIjMfkpDQYCk4xb5lq+17yajZTqNnWmnkdNKaizi8KMRsWiYJVFl0ckytSUIEAgQdx6xNuYsfEOnTpYuwbLRIfLBO00aSW0K2o40VGXv5FGASAZmbfhbkl3LSggzIDY4GceRlaP4WPsOAWJsHrmOIItOFgRIM1q8uL6IKC4LqchCB8ud+2bhG6YCJGqZSMOGDWj+4hV0wIjpdWCZ+qE78vKYkzpq8A85S92QWfWrSvdNuRDi0tUmUT4EAaJrSWvCYwYkHbfIt6QdCJ0kpU4hFSU6jZ2NxiqpscAMiI412AmbRSfL1JZ8+ZbJd+o0QlF+Ic0jOlh2/CAqlix8Q6dOjqvnOQ4eqS08bTduDgHizkxyG3PIAsRUiEv7WhAg9swfAsQeS5J2IHSSlDoFBIgO1XyHzaKTZdKx59Ly5Vsm3wkBUvl+lYVv2BAgpctEotoDTke6qTbkkkwa1IJAd1d6IQuQqP0acUJcKlai/Ej6LpeCTnuSZNc2SjU034AAsVGq/8Xhq5MkFRtxjZD0fR3nSZrJ0G3YQ3MUi2aSeVS6ZaH7wb78wNQ+bX8nBIiupYQXPgvf8ClApJtqwyuZtV+U1FFD2+Gu9EIXIFIhLhUrPmcck+zaRqmG5hsQIDZKFQIklmJapwrNUSyaSeZRZdHJCk0suJhVgQDJ3LSNPyAL3/AtQCSbao1BOowgqU1B2+EOfjUJEIlYkQ7Wxg326rR7SXZto1RD8w0IEBulCgECAWLRjlxHlUUnS6cijsq/7dkKCBDXVlaZ8WfhG5UiQFyc8pjGSpI6aqF1stLkMdR3IEDkd/ZwGeq0e0l2bcMmQvMNCBAbpQoBAgFi0Y5cR5VFJ0unIoYAWXtyURS30BoR1/bqM/4sfKNSBEjU3SI6Nz3bKsekjhr8wxbp+vFAgECA2LQuCBCLNH2N0ppOC0rfN+00MtqkxiIOPxoRi4ZZElUWnSxTW/LlWybfiSVY7mzWV8w2fSNqxiDueFxpnWwSLk7oRNm89NtN/CVtmSa1KWg70pJNfg8CBAIk2UrkISBA5KwSQ/rqJLlohKTnakdBKDc1v2zZStpoo3VJt2FHI5JobqkD6JaFbkK+/MC082P7OyFAdC0lvPAS35AuRYqaMYg7lcekTpe+qyNApN+u44NSbklWAQGSRMjd7xAgegJE59S5JLuOK9U4v+Lwq1bV0O+/r6h9NbR+FQSIRV+13aHhT6uEY3jLTc1zhQUBYtHILEQl6WSZJOPLD3Q6P1H5sf2dECAmVhPGuxLfkC5FktbdccJAKiyk4XQFSNrL3+JKUsotyRKSOmqhdbKS8lNJv0OA6AkQnVPnkuxax684bNTyyNB8AwLEovfb7tBUkgApbawKnUMIEIsGZikqSSfLJClffgABYlJKeDeKgMQ3pPYNAVKXsJRbkmUmddRC62Ql5aeSfocA0Rcg0lPnkuy6nAAp7X9x2ErYPwgBYtH7bVWwxZ8kbcRMR8FMlmCVyzcEiEUDsxSVpJNlkpQvP4AAMSklvAsBsoaA9GI1kzYibjAtjQ8nddQgQNz5NgQIBIhN64IAsUhT2vHSWbMHAWKxgBCVIgABssYQpP4qNRsswZKSCjecxDekdiOtu5mGdADJJFy5dPg3HqktPFF7VSBAwrVbX18GASL31Th/K3foQ6NGDdW+DX74wAp+ampW1xZv3N/mL15BB4xYe3oiv4AZEIFXrFq1iu666y56+OGH6aeffqItttiCjjvuOOrfvz81atRIEEN0kFAcJW50aVXN6joVPucias2etBGTNkw6jZ10dAozIKnNtOyLrnxD0skyyZG0g6aTRiXECQGiU6JmYbP0DaktSutunTpZWs/HXYxm8j4EiJnN+nzblX+E0q8yseOs/S2qXxV36ENpP5EHBqL+BgGS0ruGDRtGEydOpGOOOYY6d+5Ms2fPpkcffZROPPFEGjp0aMpYiUJyFMnoUpxilTZiUofUcT4IkNTmZ+VFV74BAbKmeKQdSWlhQoBISZmH8+Ub0uNoTZYx6dTJ0no+SwESN8MfdQSxtI0pthgswUr2H1f+EVK/Ku1N5ln7m0ldIfX/uP5kaMsTM12CNXfuXOrRowf17duXBg8eXOtVV111FU2YMIGmTp1KHTp0SPa2iBBJjmLrSMDipE3EQl4EiJR7aI6SyggNXnLpG8UCRGc5oDQ7tjv2LsSCizghQKQWYhbOl28UbIQXQEiWJ0kGmnwJA1/pSEdzmU3Usi4IEDNfiHrbpX8k9avs5ybbk0ClHX5df0tbV0i/BwJEYImjR4+mMWPG0AsvvEBt27atfWP+/PnUtWtXGjRoEJ133nmCmOoHKXaUdddtSry2rvjh0ZioqSwOU7zmrvBO6XnKUR8FAVL/BufSTejSoxjzLkBc+kaxAIkqD7ZtkxuOIUDka3FLyyLuNLnFi5fVVjnwDXftRunsoEmdrtNZkIY1CccGZPJ+1LvS0dy4tKUCpHSgpLBWPqpdhn+484+kfhWXc2mZlJsRi+pvlb5v2wdd+IGuAEk7eyP1XwgQgWzgfR5z5syhN954o17offfdl7bffnu1PyTNU+woG2zQnNKOYul0xkwcJc5gTOI0dQpp46CzB0TaOc17I+LSN5I6vXG2KPVDaRlL4+NwlRAnZkB0SjR9WF++EWd30k6ANFyld4j4+5NmiEwFiHTgitNB25FdvypuH2tp/4vLKWo/Q2j7YKU+bNrXMkknLm1sQk9oY7p3705NmjRRez5Kn169etG///5L06ZNS9VSLV++klq0aEp//vkXNW++jhIgC5eurI2r9YbN1AxI0t/4hU3Xb0qNGjaInBkp/rjCrEpSnFFpx6VjEmdcOtK8m+S78O7q1aupYcOGVFNTQ6tXrznZoZQ7h+XzHvjW9MLD4dZbr1mqsq+Gl1z6BvtEYZaPZwdLfUPH5qNYx5WxxIfiyq4S4oz6xjiWDRqwL6z1Cw4H35B5ri/f4K+J8g9p/SkNx+lIw5qE85WOTrtj2saUthsFP0LbkU2/Kqo84+rFKDuRvl8JfpC1v1VCvyrTPSC8zKpVq1Y0adKkei1Pnz59aNGiRfT888/LWiWEAoEqIgDfqKLCRFasEoBvWMWJyKqMAPyjygq0irOTqQBxOZJVxWWGrOWAAHwjB4WMLKYiAN9IhQ0v5YQA/CMnBV0F2cxUgLhcy1sFZYMs5JgAfCPHhY+slyUA34CBgEA8AfgHrKNSCGQqQEaNGkVjx451cgpWpRQAvhMEogjAN2AXIBBNAL4BywCBeALwD1hHpRDIVIDwCVg9e/aMvQdkypQp1LFjx0phie8EAWsE4BvWUCKiKiMA36iyAkV2rBKAf1jFicgcEshUgHC+hgwZQpMnT653E/rxxx9PV155pcOsI2oQCJsAfCPs8sHXZUcAvpEde6QcPgH4R/hlhC8kylyA8FG748aNo4cffpgWLFhAm2++OR133HE0YMAAaty4McoIBHJLAL6R26JHxhMIwDdgIiAQTwD+AeuoBAKZC5BKgIRvBAEQAAEQAAEQAAEQAAEQsEMAAsQOR8QCAiAAAiAAAiAAAiAAAiAgIAABIoCEICAAAiAAAiAAAiAAAiAAAnYIQIDY4YhYQAAEQAAEQAAEQAAEQAAEBAQgQASQEAQEQAAEQAAEQAAEQAAEQMAOAQgQOxwRCwiAAAiAAAiAAAiAAAiAgIBA1QiQjz76iB5//HF6++236fvvv6cWLVrQdtttR6effjrtu+++AhQI4pLAW2+9RaeeeqpK4rnnnqOtt97aZXKIuwIIrFq1iu666y51BPdPP/1EW2yxhTqCu3///tSoUaPEHHz++ec0YsQIevfdd1XY3XffnS666CJq37594ru2A5jWP5deeik99thjkZ/1yiuvUOvWrW1/MuILjMCyZcvo7rvvpk8++YQ+/vhjWrhwIfXq1Yuuu+66el+q4zs6YX0ikeZX17dCza9PtqZp/fLLL3TrrbcS1z2//vortWzZknbeeWd1N1urVq1qo3/ooYdo/Pjx9O2339LGG29MRx99NJ199tnUrFkz00+oqvfL1e+c0XPPPZfOOOMMlec82W/VCJBzzjmHZs6cSYcddhjtsMMOtHz5cnr00UeJOylDhw6lE088saoMupIy8/fff1OPHj3o559/VuUCAVJJpefuW4cNG0YTJ06sdwkp+yr7bLnnm2++oWOPPZY23HBD6tevnwrKDeHSpUuVoNlmm23cfXhEzKb1T6GBuv7666lBgwZ1UuA6rXnz5l7zg8T8E+CBs0MOOYQ23XRT2nHHHWn69OmxAkTHd3TC+sy1NL+6vhVqfn2yNUmL69a+fftS06ZNlf3x4MfixYvp/fffp4svvri2br3zzjtp5MiRymYPPPBAmjdvHk2YMIH2339/Gjt2rMknVN277733Hn333Xf18sVtFg82cF+V+6385Ml+q0aA8CjoTjvtROuss05tIf/111+q4/vbb7/Rm2++iYsNM3Jrrozuu+8+NTrC/4YAyaggAkp27ty5yje5oRs8eHDtl1111VWqEZs6dSp16NAh9ou5U/Laa6/RU089pWZO+OFZlKOOOko1gDfddJPX3JrWPwUBwqPfuIDVa9EFkxgP1HBbxZfx8kVy3CGJmgHR8R2dsL5BSPOr41sh59c33zTprV69Ws1C19TU0P3330/rrrtuZDQsSA466CC1uuT222+vDXPvvffStddeqwQIixI88QRWrFhBXbp0oS233JKmTZumAubNfqtGgMQVM09f33PPPfTyyy/XdlTgFP4I/PDDD9StWze6/PLL6ccff6RbbrkFAsQf/mBTGj16NI0ZM4ZeeOEFatu2be13zp8/n7p27UqDBg2i8847L/L7eenGXnvtpeyKZwyKn0suuUSJEl6KGdd4+oQirX8KAoRHw1auXKmWkDZs2NDnpyKtgAiUEyA6vqMTNsvsl8uvTtteKfnNknW5tAtLpbluZoHBdRHPyBYP7PL7vPSK23QeUNx7771ro+RONdfNhx56KN1www2hZjOI7+ItA7xkmNus0047TX1T3uy36gXI+eefT88++yzNmjVLNep4/BLgdY08WjJp0iQlPiBA/PIPNTXe5zFnzhx644036n0ij6ptv/32an9I1MPT2X369FFT1SeccEKdIA8++CBdccUVNHnyZNp1110zz760/ikIEBZNLLB4DfUBBxygGqh27dplng98gF8C5TrkOr6jE9ZvDuumlkaARPlWpeQ3S9bl0v7f//6n6l2e/Rg1ahRxXcsCZJdddiGuo3bbbTf1+pAhQ1Qd+8EHH9Tb78EzKLwU9plnngk1m0F8F4uOGTNmqH02hX01ebPfqhYgX375pVrmwUr+5ptvDsLo8vQRvIb5zDPPVGvyeU0zlwEESJ4sID6v3bt3pyZNmqi1r6UPLzvhDklhWrr0dx5Q4CVYPPV/8MEH1/n5xRdfVDbHS7AOP/zwTGHr1D+8lrrQCePRRl5vzUvRWJA88sgjapoeT34IlOuQ6/iOTtgs6eoKkDjfqpT8Zsm6XNpcd3IdutFGG9Eee+yhZpl5Q/ptt91GPLvBbTkvjeUZahYn3IEufc466yw1sMS/44kmsGDBArVEjQeZivfL5M1+q1aAsAI//vjj1UkivJ68TZs28AWPBHj/DVdevMaRT87gBwLEYwEEnhQvs+JRH54ZK314dmPRokX0/PPPR+ZiypQpatqaR+r222+/OmFef/11dYoWj+Tx4ENWj436hxtxHiWLOwkpq7whXfcEynXIdXxHJ6z7XMWnoCNAyvlWpeQ3S9bl0uaTKnkZ1j777EO8n6PwvPPOO3TSSSfREUccQTfeeCOdcsopxCKQ69vS54ILLqCnn36aPv3001Czmfl33XHHHWqJGrNkpoUnb/ZblQKEO7/cCeHj+8aNG0d77rln5gaXtw/gtYx8whFPw/LxfBAgebOA8vk1GekJfQbEZv1zzDHHqBHIqIYeFlW9BDADEl22Sb5lUq9UrzXJc8YzG7xygfeu8cBH8cOzzcyfD/TBDIicaVRIPiyFjzfmer14f03e7LfqBAifrMH7DnhqkEfcefkVHr8EeHqRlTyPkvB60MLD60r5Hx5Z2WqrrepsPvb7hUgtawIma11D3gNiu/7h5Qx8gAZvTseTHwLYA1K/rCW+ZVKv5Me64nNa2NvBR+zy8qDip3fv3mpWg+si7AFJby0ffvih6hdFHTefN/utKgHClTavDWcFz2uqeQkQHv8EPvvsM+rZs2fZhPlAAKwR9V82oaTIGxx57auLU7CefPJJNQDh+xQsF/UPj4j98ccfaqMinvwQKCdAdHxHJ2yWdJOWYEl9q1LymyXrcmnzHg8+Fp2PQy8ePOR3CoLk1VdfVRvQWYSUnoLFMyS84gSnYMVT5iXpDzzwgNpPw5c7Fj95s9+qESB8bjWvPeQjOIcPH06s1vFkQ4DX6EadbsTrQnlJFh/fx5cb8SwJnnwS4BOwWKTG3QPC+zw6duxI//zzj7rAaf3116fNNtusFhbftsvT12xThVvCC/eA8L4Q34dOSOufqPzw5Zx890fpUZdcl/FRxLyXrbCPKp/Wkr9cl+uQS32HqemEzZJyufxKfauS8psl63Jp84mVvNSqffv2agl1o0aNVHAe1OVlV3z569VXX6326PHqEq5reYN64SncAxJ1QEioefb5XTyLx/dUbbLJJqqvWvpUir/aYlY1AoQvv2HjZ/XNTlL68GbowlFntuAhHj0C2ISux6vaQxem8XmfQ+fOnWn27NnqVKziDnfhtuTSjdhfffWVGqFr2bJl7U3ovLxvyZIlamRp22239YpPWv9E5YdnDAcOHKgE+dZbb61OB+NTsJ544gl1dxGfuY+6y2txZpYYn3zGM17c6eb6slOnTmo0mR/uGLIo50fiO4VM6IT1nXFJfqW+VQn59c03TXo8q3HNNdeoU7COPPJI4iXVXLc2b95c1c+Fi18LG6m53iq+CZ03sPPeWzz1CfAlzDx4xoPlXOdHPSH7q+0yrRoB0q9fP5o5c2YsH77yni/IwZMdAQiQ7NiHmDKPenJDxYKBGzm+AZpFxYABA2pvA48TIJwfHi0aMWKEEi787L777nThhRfWdtJ85lla/0Tlh0/q402fvLaaN5zzLAk38jzCyKOOhUMcfOYHaWVDgEUGX94a9XBHnMU6PxLfKcShE9Z3riX5lfpWJeTXN9+06fHJoTygO2/ePCU8+G4m7jQXXxrLcfMphty34llqrqd42Tsvg+d38NQnwPuTeU8f/8PtXdQTsr/aLtOqESC2wSA+EAABEAABEAABEAABEAAB+wQgQOwzRYwgAAIgAAIgAAIgAAIgAAIxBCBAYBogAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABCBAYAMgAAIgAAIgAAIgAAIgAALeCECAeEONhEAABEAABEAABEAABEAABP4PdDyWecu+S/8AAAAASUVORK5CYII=" width="640">


We can see a couple of things in these histograms: 
- It does not look like any of the numerical features were clipped of at any maximum or minimum value (which is great so we won't have to fix that).
- The features are on very different scales, so we will have to rescale them later on.
- Some of the features are quite skewed, which can be a problem for some algorithms. We could try to transform them into a more normally distributed shape later.
- The \`compression_ratio\` seems to cluster into a low-ratio and a smaller high-ratio group. Since the data set is quite small, we should probably stratify for that when we split off a test set to avoid introducing sampling bias.

We can now put aside a hold-out test set.


\`\`\`python
# assign each instance a low (0) or high (1) compression_ratio category for stratification
auto["cr_cat"] = pd.cut(auto.compression_ratio,2,labels=[0,1])

from sklearn.model_selection import StratifiedShuffleSplit
# put 20%  of the data aside as a test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split.split(auto, auto["cr_cat"]):
    auto_train_set = auto.loc[train_index]
    auto_test_set = auto.loc[test_index]
    
# remove the cr_cat from the data again
for set in (auto_train_set, auto_test_set):
    set.drop(["cr_cat"], axis=1, inplace=True)
\`\`\`

Now we put a test set aside to protect us from snooping, we will do some exploration on a copy of our training data. Let's look for some (Pearson) correlation first.


\`\`\`python
# calculate correlations with price
auto = auto_train_set.copy()
auto.corr().price.sort_values(ascending=False)
\`\`\`




    price                1.000000
    engine_size          0.872527
    curb_weight          0.866972
    horsepower           0.808510
    width                0.791171
    length               0.736342
    wheel_base           0.648699
    bore                 0.543084
    normalized_losses    0.189377
    height               0.147473
    stroke               0.076966
    compression_ratio    0.031453
    symboling           -0.116180
    peak_rpm            -0.120205
    city_mpg            -0.697621
    highway_mpg         -0.725255
    Name: price, dtype: float64



We can also do that graphically for a few examples with Pandas' super cool scatter matrix plot:


\`\`\`python
# plot a scatter matrix
attributes = ["price","engine_size","compression_ratio","highway_mpg"]
pd.plotting.scatter_matrix(auto[attributes])
plt.tight_layout()
\`\`\`


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="640">


It's clear that e.g. engine size and price are positively correlated, compression ratio and price don't seem to be correlated much, and mileage and price are negatively correlated.

At this point we could think about creating some combined features, calculated from the ones we got, or adding some polynomial features (e.g. price vs. mileage doesn't look perfectly linear. But it's a good idea to keep it simple first and engineer more features later on if necessary.

We should now come up with a strategy for th missing values.


\`\`\`python
# show sum of missing values per attribute
auto.isnull().sum()
\`\`\`




    symboling             0
    normalized_losses    29
    make                  0
    fuel_type             0
    aspiration            0
    num_of_doors          2
    body_style            0
    drive_wheels          0
    engine_location       0
    wheel_base            0
    length                0
    width                 0
    height                0
    curb_weight           0
    engine_type           0
    num_of_cylinders      0
    engine_size           0
    fuel_system           0
    bore                  4
    stroke                4
    compression_ratio     0
    horsepower            1
    peak_rpm              1
    city_mpg              0
    highway_mpg           0
    price                 4
    dtype: int64



The instances with missing price are not useful for us so we will drop them. We will also separate the price (our labels) from the features.


\`\`\`python
# drop rows with missing price
auto.dropna(subset=["price"], inplace=True)
# separate labels and features
auto_labels = auto.price.copy()
auto_feat = auto.drop(["price"], axis=1)
\`\`\`

We can drop the instances with the other missing values as well, but since we only have a few instances in this data set it might be a good idea to try to fill the missing values (e.g. with the feature's median for numeric features or its mode for categorical features) instead. We can treat this choice (dropping or filling of missing values) as a hyper-parameter for our model. For the missing \`num_of_doors\` we will fill in the majority value. For numerical features, Scikit-Learn offers an Imputer which can be used to fill the value (e.g. median) for the training set into missing values in the test set as well.


\`\`\`python
# get the majority number of doors and use it to fill missing values
num_d = auto_feat.num_of_doors.value_counts().sort_values(ascending=False).index[0]
auto_feat.num_of_doors.fillna(num_d, inplace=True)

# import the imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# extract the numerical features and fill their missing values
auto_featnum = auto_feat[[c for c in auto_feat.columns if auto_feat[c].dtype!="O"]]
imputer.fit(auto_featnum)
auto_featnum_filled = pd.DataFrame(imputer.transform(auto_featnum),
                                   columns=auto_featnum.columns)
\`\`\`

We also have to scale the numerical features before as they are on very different scales. We'll used standardization here.


\`\`\`python
# standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
auto_featnum_scaled = scaler.fit_transform(auto_featnum_filled)
\`\`\`

Since the algorithms won't be able to work with the categorical data directly, we'll have to encode the categorical features first. \`num_of_door\` and \`num_of_cylinders\` are ordinal (meaning that they have a natural order) but maybe not evenly spaced and the other categorical features' values have no natural order so the simplest solution is to use One-Hot encoding for all categorical features.


\`\`\`python
# import label binarizer for 1-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# extract the categorical features
auto_featcat = auto_feat.drop([c for c in auto_feat.columns if auto_feat[c].dtype!="O"],axis=1)
auto_featcat_1hot = encoder.fit_transform(auto_featcat)
\`\`\`

At last, we can combine the features again to get one feature matrix.


\`\`\`python
# recombine features
auto_feat_prepro = np.concatenate([auto_featnum_scaled, auto_featcat_1hot],axis=1)
\`\`\`

Finally done with the preprocessing! Or are we? We've done everything manually so far but actually we can write that all up more efficiently by transformer for our custom transformations and put them in pipelines with Scikit-Learn transformers!


\`\`\`python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# a transformer class to select features from a data frame
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names].values

# a transformer class to fill missing num_of_doors
class NumDoorsFill(BaseEstimator, TransformerMixin):
    def __init__(self, fill_na=True):
        self.fill_na = fill_na
    def fit(self, X, y=None):
        num_d = X.num_of_doors.value_counts().sort_values(ascending=False).index[0]
        return self
    def transform(self, X, y=None):
        if self.fill_na:
            X.num_of_doors.fillna(num_d,inplace=True)
            return X
        else:
            return X.dropna(subset=["num_of_doors"], inplace=True)
    
# extract numerical and categorical feature names
num_feat = [c for c in auto_feat.columns if auto_feat[c].dtype!="O"]
cat_feat = [c for c in auto_feat.columns if auto_feat[c].dtype=="O"]

# pipeline for the numerical features
num_pipeline = Pipeline([
               ('selector', DataFrameSelector(num_feat)),
               ('imputer', SimpleImputer(strategy="median")),
               ('std_scaler', StandardScaler())])

# pipeline for the categorical features
cat_pipeline = Pipeline([
               ('selector', DataFrameSelector(cat_feat)),
               ('1hot_encoder', OneHotEncoder()),])

# combination of both pipelines
full_pipeline = FeatureUnion(transformer_list=[
               ("num_pipeline", num_pipeline),
               ("cat_pipeline", cat_pipeline),])

auto_feat_prepro = full_pipeline.fit_transform(auto_feat)

\`\`\`

Now we're really done, let's train some models.

#### Selecting and tuning a  model
Before we get into fine tuning a model we will first spot-check a couple of commonly used algorithms with default parameters to pick the most promising one for further optimization. We'll choose some simple algorithms here (linear regression, K-neighbors regression, and support vector regression). If this was a real task we would probably try more algorithms and ensemble methods and short-list a couple of promising candidates.


\`\`\`python
# import cross validation and some models (linear reg, KNN, SVR)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# cross-evaluate a linear regression model
linreg = LinearRegression()
linreg_scores = cross_val_score(linreg, auto_feat_prepro, auto_labels,scoring="r2", cv=10)

# cross-evaluate a K-Neighbors regression model
knreg = KNeighborsRegressor()
knreg_scores = cross_val_score(knreg, auto_feat_prepro, auto_labels,scoring="r2", cv=10)

# cross-evaluate a suport vector regression model
svreg = SVR(kernel="linear",gamma="scale")
svreg_scores = cross_val_score(svreg, auto_feat_prepro, auto_labels,scoring="r2", cv=10)

print("Cross-evaluation score for {}: {}+/-{}".format("LR",linreg_scores.mean(),linreg_scores.std()))
print("Cross-evaluation score for {}: {}+/-{}".format("KNR",knreg_scores.mean(),knreg_scores.std()))
print("Cross-evaluation score for {}: {}+/-{}".format("SVR",svreg_scores.mean(),svreg_scores.std()))
\`\`\`

    Cross-evaluation score for LR: 0.8056458770764865+/-0.1194569502992223
    Cross-evaluation score for KNR: 0.7455563483057739+/-0.16928604290397686
    Cross-evaluation score for SVR: 0.060918559892998804+/-0.15163455291086678
    

Looks like the linear regression looks most promising in this example. We will pick this model for further fine-tuning. The standard linear regression, however, does not have any hyper-parameters to tune, but we can introduce different kinds of regularization approaches (Lasso and Ridge).


\`\`\`python
# import grid search and regularized linear models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# values for the regularization parameter alpha
param_grid = {"alpha": [0.01,0.1,1,10,100]}

# lasso-regularized linear regression
grid_lasso = GridSearchCV(Lasso(max_iter=100000), param_grid=param_grid, cv=10,scoring="r2")
_ = grid_lasso.fit(auto_feat_prepro, auto_labels)

# ridge-regularized linear regression
grid_ridge = GridSearchCV(Ridge(max_iter=100000), param_grid=param_grid, cv=10,scoring="r2")
_ = grid_ridge.fit(auto_feat_prepro, auto_labels)

# print results
print("Best score for lasso regression:\\n {}+/-{} for {}".format(grid_lasso.best_score_,grid_lasso.cv_results_["std_test_score"][grid_lasso.best_index_],grid_lasso.best_params_))
print("Best score for ridge regression:\\n {}+/-{} for {}".format(grid_ridge.best_score_,grid_ridge.cv_results_["std_test_score"][grid_ridge.best_index_],grid_ridge.best_params_))
\`\`\`

    Best score for lasso regression:
     0.8625877052058882+/-0.07134551312938564 for {'alpha': 10}
    Best score for ridge regression:
     0.866055991114397+/-0.06601838158702997 for {'alpha': 1}
    

Cool, we managed to improve the *R<sup>2<sup>* cross-validation score of our model by 6% by using regularization. Now we can finally tackle the test set!



\`\`\`python
from sklearn.metrics import r2_score

# get the test set
test = auto_test_set.copy()

# drop rows with missing price
test.dropna(subset=["price"], inplace=True)

# separate labels and features
test_labels = test.price.copy()
test_feat = test.drop(["price"], axis=1)

# preprocess the test set
test_feat_prepro = full_pipeline.fit_transform(test_feat)

# fit the ridge model
ridge_regression = Ridge(alpha=1).fit(test_feat_prepro, test_labels)

# make predictions for the test set
predictions = ridge_regression.predict(test_feat_prepro)

# evaluate the model
test_score = r2_score(test_labels,predictions)
print("Test score: "+str(test_score))
\`\`\`

    Test score: 0.9843494555106602
    

Wow, that turned out exceptionally good. Such an improvement from cross-validation to test score is unusual; in fact, the test score is usually lower. We were probably lucky with the few examples that ended up in our test set. The whole data set is fairly small after all. We could check some more models now, test some more feature modifications or add additional features (e.g. different degrees of polynomial features) but I think everything so far serves as a good example already.

### Ensemble methods for Pulsar Neutron Star Classification <a id="neutronstar"></a>
As a former scientist I love pretty much everything in physics and astronomy has always interested me due to the extreme conditions being studied. Some of the most fascinating objects  found in space are Pulsars, a special kind of neutron stars. You can find the Jupyter Notebook and data [here](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository/tree/master/Ensembles-for-neutron-stars). We can read in the description accompanying this data set: 

"Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. \\[...\\] As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars
rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes." 

This data set contains events of radio signal detection representing Pulsar candidates, which were recorded during the [High Time Resolution Universe Survey 1](https://academic.oup.com/mnras/article/409/2/619/1037409). It was contributed to the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/HTRU2#) by Robert Lyon from the University of Manchester's School of Physics and Astronomy.

#### Frame the Problem

To detect the Pulsars, the signal is recorded over many rotations of the pulsar and the integrated pulse profile is used along with the DM-SNR curve (dispersion measure - signal to noise ratio curve, see [here](http://www.scienceguyrob.com/wp-content/uploads/2016/12/WhyArePulsarsHardToFind_Lyon_2016.pdf)).
However, Pulsars are very rare and many spurious events caused by radio frequency interference (RFI) get detected, which makes it hard to find legitimate signals. It is very time-intensive and costly to have every record checked by human experts, therefore, machine learning could be of great aid to Pulsar detection.

We have a two-class classification problem at hand (Pulsar or not Pulsar) and we are most likely dealing with a high class imbalance because Pulsars are very rare. Because of this imbalance, we should not choose accuracy as a metric to evaluate the performance of our model. It is probably not a catastrophe if human experts will still find some false positives which they have to filter out manually, but we'd be really sad if me miss a rare Pulsar (false negative). Hence, we should optimize for recall to minimize the false negative ratio.


\`\`\`python
# import libraries 
import numpy as np # numerical computation
import pandas as pd # data handling
import warnings
warnings.filterwarnings('ignore')
# visulalization
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("darkgrid")
%matplotlib notebook
\`\`\`

#### Data preparation
Let's import the data into data frame hand have a look.


\`\`\`python
# load data and show first 5 rows
# the columns are:
# 1. Mean of the integrated profile.
# 2. Standard deviation of the integrated profile.
# 3. Excess kurtosis of the integrated profile.
# 4. Skewness of the integrated profile.
# 5. Mean of the DM-SNR curve.
# 6. Standard deviation of the DM-SNR curve.
# 7. Excess kurtosis of the DM-SNR curve.
# 8. Skewness of the DM-SNR curve.
# 9. Class (1 = Pulsar, 0 = no Pulsar)

pulsar = pd.read_csv("HTRU_2.csv",header=None,names=["ip_mean","ip_std","ip_kurt","ip_skew","dmsnr_mean","dmsnr_std","dmsnr_kurt","dmsnr_skew","label"])
pulsar.head()
\`\`\`




<div class="post-page-table">
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
      <th>ip_mean</th>
      <th>ip_std</th>
      <th>ip_kurt</th>
      <th>ip_skew</th>
      <th>dmsnr_mean</th>
      <th>dmsnr_std</th>
      <th>dmsnr_kurt</th>
      <th>dmsnr_skew</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140.562500</td>
      <td>55.683782</td>
      <td>-0.234571</td>
      <td>-0.699648</td>
      <td>3.199833</td>
      <td>19.110426</td>
      <td>7.975532</td>
      <td>74.242225</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>102.507812</td>
      <td>58.882430</td>
      <td>0.465318</td>
      <td>-0.515088</td>
      <td>1.677258</td>
      <td>14.860146</td>
      <td>10.576487</td>
      <td>127.393580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103.015625</td>
      <td>39.341649</td>
      <td>0.323328</td>
      <td>1.051164</td>
      <td>3.121237</td>
      <td>21.744669</td>
      <td>7.735822</td>
      <td>63.171909</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>136.750000</td>
      <td>57.178449</td>
      <td>-0.068415</td>
      <td>-0.636238</td>
      <td>3.642977</td>
      <td>20.959280</td>
      <td>6.896499</td>
      <td>53.593661</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88.726562</td>
      <td>40.672225</td>
      <td>0.600866</td>
      <td>1.123492</td>
      <td>1.178930</td>
      <td>11.468720</td>
      <td>14.269573</td>
      <td>252.567306</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




\`\`\`python
# print info
pulsar.info()
\`\`\`

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17898 entries, 0 to 17897
    Data columns (total 9 columns):
    ip_mean       17898 non-null float64
    ip_std        17898 non-null float64
    ip_kurt       17898 non-null float64
    ip_skew       17898 non-null float64
    dmsnr_mean    17898 non-null float64
    dmsnr_std     17898 non-null float64
    dmsnr_kurt    17898 non-null float64
    dmsnr_skew    17898 non-null float64
    label         17898 non-null int64
    dtypes: float64(8), int64(1)
    memory usage: 1.2 MB
    


\`\`\`python
# print describtion 
pulsar.describe()
\`\`\`




<div class="post-page-table">
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
      <th>ip_mean</th>
      <th>ip_std</th>
      <th>ip_kurt</th>
      <th>ip_skew</th>
      <th>dmsnr_mean</th>
      <th>dmsnr_std</th>
      <th>dmsnr_kurt</th>
      <th>dmsnr_skew</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
      <td>17898.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>111.079968</td>
      <td>46.549532</td>
      <td>0.477857</td>
      <td>1.770279</td>
      <td>12.614400</td>
      <td>26.326515</td>
      <td>8.303556</td>
      <td>104.857709</td>
      <td>0.091574</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25.652935</td>
      <td>6.843189</td>
      <td>1.064040</td>
      <td>6.167913</td>
      <td>29.472897</td>
      <td>19.470572</td>
      <td>4.506092</td>
      <td>106.514540</td>
      <td>0.288432</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.812500</td>
      <td>24.772042</td>
      <td>-1.876011</td>
      <td>-1.791886</td>
      <td>0.213211</td>
      <td>7.370432</td>
      <td>-3.139270</td>
      <td>-1.976976</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>100.929688</td>
      <td>42.376018</td>
      <td>0.027098</td>
      <td>-0.188572</td>
      <td>1.923077</td>
      <td>14.437332</td>
      <td>5.781506</td>
      <td>34.960504</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>115.078125</td>
      <td>46.947479</td>
      <td>0.223240</td>
      <td>0.198710</td>
      <td>2.801839</td>
      <td>18.461316</td>
      <td>8.433515</td>
      <td>83.064556</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>127.085938</td>
      <td>51.023202</td>
      <td>0.473325</td>
      <td>0.927783</td>
      <td>5.464256</td>
      <td>28.428104</td>
      <td>10.702959</td>
      <td>139.309331</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>192.617188</td>
      <td>98.778911</td>
      <td>8.069522</td>
      <td>68.101622</td>
      <td>223.392140</td>
      <td>110.642211</td>
      <td>34.539844</td>
      <td>1191.000837</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




\`\`\`python
# look for missing values
pulsar.isna().sum()
\`\`\`




    ip_mean       0
    ip_std        0
    ip_kurt       0
    ip_skew       0
    dmsnr_mean    0
    dmsnr_std     0
    dmsnr_kurt    0
    dmsnr_skew    0
    label         0
    dtype: int64




\`\`\`python
# look at class distribution
pulsar.label.value_counts()
\`\`\`




    0    16259
    1     1639
    Name: label, dtype: int64



We have 17898 instances of which only 9.2 % correspond to the positive class. The 8 numerical features correspond to simple statistics (mean, standard deviation, excess kurtosis, and skewness) of the radio signals' integrated profile and DM-SNR curve.

We can have a look on how the features are distributed by plotting them in histograms:


\`\`\`python
# plot histograms
pulsar.hist()
plt.tight_layout()
\`\`\`


<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/neutronstar_output_10_0.png">


We can see a couple of things in these histograms: 
- It does not look like any of the features were clipped of at any maximum or minimum value (which is great so we won't have to fix that).
- The features are on very different scales, so we will have to rescale them later on.
- Some of the features are very skewed or have outliers, which can be a problem for some algorithms and scaling transformations. We could try to transform them into a more Gaussian shape.

We can now put aside a hold-out test set. Or actually, since I want to work with ensemble methods in this example, I will also split the train set into two subsets. One will be used to train and tune different classifiers and the second one will be used to train another classifier in a second layer which will blend the outputs of the first layer of classifiers. If we are lucky, our blended ensemble will work better than the individual classifiers.


\`\`\`python
from sklearn.model_selection import StratifiedShuffleSplit
# put 20%  of the data aside as a test set
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split1.split(pulsar, pulsar["label"]):
    pulsar_train = pulsar.loc[train_index]
    pulsar_test = pulsar.loc[test_index]
    X_test = pulsar_test.loc[:,"ip_mean":"dmsnr_skew"]
    y_test = pulsar_test.loc[:,"label"]
    
    
# put 25%  of the training data aside for the blender layer 2
pulsar_train = pulsar_train.reset_index(drop=True)
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1234)
for train_index, test_index in split2.split(pulsar_train, pulsar_train["label"]):
    pulsar_layer1 = pulsar_train.reindex().loc[train_index]
    X_layer1 = pulsar_layer1.loc[:,"ip_mean":"dmsnr_skew"]
    y_layer1 = pulsar_layer1.loc[:,"label"]
    
    pulsar_layer2 = pulsar_train.reindex().loc[test_index]
    X_layer2 = pulsar_layer2.loc[:,"ip_mean":"dmsnr_skew"]
    y_layer2 = pulsar_layer2.loc[:,"label"]

\`\`\`

Let's see if  we can bring the data in a more Gaussian shape with Scikit-Learn's \`PowerTransformer\`. If there will still be strong outliers which would cause problems we could also try a \`QuantileTransformer\` which would force all values in a similar range and still produce a kind of Gaussian shaped distribution. At this point I have no idea which may work better so we'll just treat it as a hyper-parameter and try both. Let's write a transformer class for it.


\`\`\`python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# a transformer class to select either PowerTranformer or gaussian QuantileTransformer
class PQTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, trans="power"):
        self.trans = trans
        self.pt = PowerTransformer()
        self.qt = QuantileTransformer(output_distribution="normal")
        
    def fit(self, X, y=None):
        self.pt.fit(X)
        self.qt.fit(X)
        return self
    
    def transform(self, X):
        if self.trans == "power":
            return self.pt.transform(X)
        elif self.trans == "quantile":
            return self.qt.transform(X)
        else:
            return None

\`\`\`

Let's have a look at the feature distribution after both transform options:


\`\`\`python
# use a PowerTransformer
transformer = PQTransformer(trans="power")
X_layer1_trans = pd.DataFrame(transformer.fit_transform(X_layer1),
                              columns=X_layer1.columns)
X_layer1_trans.plot(kind="box")
plt.tight_layout()
\`\`\`


<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/neutronstar_output_16_0.png">



\`\`\`python
# use a Gaussian QuantileTransformer
transformer = PQTransformer(trans="quantile")
X_layer1_trans = pd.DataFrame(transformer.fit_transform(X_layer1),
                              columns=X_layer1.columns)
X_layer1_trans.plot(kind="box")
plt.tight_layout()
\`\`\`


<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/neutronstar_output_17_0.png">


The transformed data looks as expected; after the \`PowerTransformer\`, the data is still on quite different scales due to the outlying values. The \`QuantileTransformer\` brings everything on the same range but introduced [saturation artifacts](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py) for extreme values. We'll see what works better.

Since all the 8 features come from only two types of measurements, they are most likely correlated in some way. Let's have a look at the correlation matrix.


\`\`\`python
# plot correlation matrix
sns.heatmap(X_layer1_trans.corr(),annot=True, cmap=plt.cm.seismic)
\`\`\`




    <matplotlib.axes._subplots.AxesSubplot at 0x7f089bf41400>




<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/neutronstar_output_19_1.png">


We can try to incorporate feature interaction by adding polynomial features, treating the degree as a hyper-parameter. However we should actually create these features before we do our power/quantile transform so we'll restructure the work flow a bit.


\`\`\`python
# put polynomial features and Transformer into a pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
preprocess = Pipeline([
               ('poly', PolynomialFeatures()),
               ('trans', PQTransformer())])
\`\`\`

#### Selecting and tuning models
Time to train some classifiers on the data. We'll start with a simple logistic regression to get somewhat of a benchmark.


\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# set up a pipeline with the preprocessing and a classifier 
lr_clf = Pipeline([
               ('prep', preprocess),
               ('lr', LogisticRegression(class_weight="balanced",
                                         solver="lbfgs",
                                         random_state=1234))]) # acount for class imbalance

# perform a grid search cross evaluation of the paramters below
param_grid = {
    'prep__poly__degree': [1,2,3],
    'prep__trans__trans': ["power","quantile"],
    'lr__C': [0.01,0.1,1,10],
}

lr_grid = GridSearchCV(lr_clf, param_grid, scoring="recall",iid=False, cv=5,n_jobs=-1)
lr_grid.fit(X_layer1, y_layer1)
print("Best parameter (CV recall score=%0.3f):" % lr_grid.best_score_)
print(lr_grid.best_params_)

\`\`\`

    Best parameter (CV recall score=0.927):
    {'lr__C': 1, 'prep__poly__degree': 1, 'prep__trans__trans': 'quantile'}
    


\`\`\`python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
# let's do a prediction for the layer 2 test set
y_pred = lr_grid.predict(X_layer2)
print("Logistic regression on layer 2 test set.\\nConfusion matrix:")
print(confusion_matrix(y_layer2, y_pred))
print("Recall: {:0.3f}".format(recall_score(y_layer2, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_layer2, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_layer2, y_pred)))
\`\`\`

    Logistic regression on layer 2 test set.
    Confusion matrix:
    [[3076  176]
     [  27  301]]
    Recall: 0.918
    Precision: 0.631
    Accuracy: 0.943
    

That already looks pretty okay. Since we were optimizing for recall the precision is lousy but thats probably okay - rather have humans dig through some spurious signals than missing a real one. But unfortunately we're still missing a few Pulsars (false negatives) here, so let's see if we can do better with applying ensemble methods. One of the most popular algorithms in that field is random forest. It basically is an ensemble of decision trees of which each just deals with a subset of features and a bootstrapped subset of training instances. The prediction of all these simple trees is then combined to come up with a final prediction of the whole forest. The forest can be regularized by regularizing its trees (e.g. limiting number of leave nodes or depth).


\`\`\`python
from sklearn.ensemble import RandomForestClassifier

# set up a pipeline with the preprocessing and a classifier 
rf_clf = Pipeline([
               ('prep', preprocess),
               ('rf', RandomForestClassifier(class_weight="balanced_subsample",
                                             random_state=1234,
                                             n_jobs=-1))]) # acount for class imbalance

# perform a grid search cross evaluation of the paramters below
param_grid = {
    'prep__poly__degree': [1,2,3],
    'prep__trans__trans': ["power","quantile"],
    'rf__n_estimators': [100,300,500],
    'rf__max_leaf_nodes': [5,10,20],
}

rf_grid = GridSearchCV(rf_clf, param_grid, scoring="recall",iid=False, cv=5,n_jobs=-1)
rf_grid.fit(X_layer1, y_layer1)
print("Best parameter (CV recall score=%0.3f):" % rf_grid.best_score_)
print(rf_grid.best_params_)
\`\`\`

    Best parameter (CV recall score=0.910):
    {'prep__poly__degree': 3, 'prep__trans__trans': 'power', 'rf__max_leaf_nodes': 10, 'rf__n_estimators': 100}
    


\`\`\`python
# we reached the border of our n_estimator parameter, so lets do another grid search around that parameter
rf2_clf = Pipeline([
               ('prep', preprocess),
               ('rf', RandomForestClassifier(class_weight="balanced_subsample",
                                             random_state=1234,
                                             n_jobs=-1))]) # acount for class imbalance
param_grid = {
    'prep__poly__degree': [3],
    'prep__trans__trans': ["power"],
    'rf__n_estimators': [50,100],
    'rf__max_leaf_nodes': [4,6,10],
}

rf2_grid = GridSearchCV(rf2_clf, param_grid, scoring="recall",iid=False, cv=5,n_jobs=-1)
rf2_grid.fit(X_layer1, y_layer1)
print("Best parameter (CV recall score=%0.3f):" % rf2_grid.best_score_)
print(rf2_grid.best_params_)
\`\`\`

    Best parameter (CV recall score=0.910):
    {'prep__poly__degree': 3, 'prep__trans__trans': 'power', 'rf__max_leaf_nodes': 6, 'rf__n_estimators': 100}
    


\`\`\`python
# let's do a prediction for the layer 2 test set
y_pred = rf2_grid.predict(X_layer2)
print("Random forrest on layer 2 test set.\\nConfusion matrix:")
print(confusion_matrix(y_layer2, y_pred))
print("Recall: {:0.3f}".format(recall_score(y_layer2, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_layer2, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_layer2, y_pred)))
\`\`\`

    Random forrest on layer 2 test set.
    Confusion matrix:
    [[3181   71]
     [  33  295]]
    Recall: 0.899
    Precision: 0.806
    Accuracy: 0.971
    

Looks like the forest actually performed (a little bit) worse than just the logistic regression. Maybe we can try another kind of ensemble, e.g. AdaBoost. In this ensemble, we subsequently train weak learners and each learner will try to correct the errors that its predecessor makes. We could use pretty much any base estimator, but let's stick to trees. Or rather stumps because we won't allow them to perform more than one split per estimator. 


\`\`\`python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# set up a pipeline with the preprocessing and a classifier 
ab_clf = Pipeline([
               ('prep', preprocess),
               ('ab', AdaBoostClassifier(
                   base_estimator=DecisionTreeClassifier(max_depth=1,
                                                         class_weight="balanced"),
                   algorithm="SAMME.R",
                   random_state=1234))]) # acount for class imbalance

# perform a grid search cross evaluation of the paramters below
param_grid = {
    'prep__poly__degree': [1,2,3],
    'prep__trans__trans': ["power","quantile"],
    'ab__n_estimators': [10,50,100],
    'ab__learning_rate': [0.5,1.0,1.5],
}

ab_grid = GridSearchCV(ab_clf, param_grid, scoring="recall",iid=False, cv=5,n_jobs=-1)
ab_grid.fit(X_layer1, y_layer1)
print("Best parameter (CV recall score=%0.3f):" % ab_grid.best_score_)
print(ab_grid.best_params_)
\`\`\`

    Best parameter (CV recall score=0.919):
    {'ab__learning_rate': 0.5, 'ab__n_estimators': 10, 'prep__poly__degree': 3, 'prep__trans__trans': 'power'}
    


\`\`\`python
# we reached the border of our n_estimator parameter, so lets do another grid search around that parameter
ab2_clf = Pipeline([
               ('prep', preprocess),
               ('ab', AdaBoostClassifier(
                   base_estimator=DecisionTreeClassifier(max_depth=1,
                                                         class_weight="balanced"),
                   algorithm="SAMME.R",
                   random_state=1234))]) # acount for class imbalance

param_grid = {
    'prep__poly__degree': [3],
    'prep__trans__trans': ["power"],
    'ab__n_estimators': [1,2,3,10],
    'ab__learning_rate': [0.1,0.3,0.5],
}

ab2_grid = GridSearchCV(ab2_clf, param_grid, scoring="recall",iid=False, cv=5,n_jobs=-1)
ab2_grid.fit(X_layer1, y_layer1)
print("Best parameter (CV recall score=%0.3f):" % ab2_grid.best_score_)
print(ab2_grid.best_params_)
\`\`\`

    Best parameter (CV recall score=0.924):
    {'ab__learning_rate': 0.3, 'ab__n_estimators': 2, 'prep__poly__degree': 3, 'prep__trans__trans': 'power'}
    


\`\`\`python
# let's do a prediction for the layer 2 test set
y_pred = ab2_grid.predict(X_layer2)
print("Random forrest on layer 2 test set.\\nConfusion matrix:")
print(confusion_matrix(y_layer2, y_pred))
print("Recall: {:0.3f}".format(recall_score(y_layer2, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_layer2, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_layer2, y_pred)))
\`\`\`

    Random forrest on layer 2 test set.
    Confusion matrix:
    [[3095  157]
     [  24  304]]
    Recall: 0.927
    Precision: 0.659
    Accuracy: 0.949
    

Nice, we minimally improved the performance compared to the logistic regression. Now how about if we could stuff our ensembles into a bigger ensemble? Each of the classifiers we have trained and tuned so far can predict a probability for a class. So we could take into account all these probabilities to find out which class is considered most likely by all the classifiers combined. Scikit-Learn offers a \`VotingClassifier\` for this purpose.  


\`\`\`python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.ensemble import VotingClassifier
# set up the preprep pipes with the classifiers from earlier with the optimized parameters

# preprosessing + logistic regression
lr_pipe = Pipeline([('poly',PolynomialFeatures(degree=3)),
                    ('trans',PQTransformer(trans="quantile")),
                    ('lr', LogisticRegression(C=1,
                                             class_weight="balanced",
                                             solver="lbfgs",
                                             max_iter=100000,
                                             random_state=1234))])
# preprosessing + random forest
rf_pipe = Pipeline([('poly',PolynomialFeatures(degree=3)),
                    ('power',PQTransformer(trans="power")),
                    ('rf', RandomForestClassifier(n_estimators=100,
                                                  max_leaf_nodes=6,
                                                  class_weight="balanced_subsample",
                                                  random_state=1234,
                                                  n_jobs=-1))])
# preprosessing + AdaBoost
ab_pipe = Pipeline([('poly',PolynomialFeatures(degree=3)),
                    ('power',PQTransformer(trans="power")),
                    ('ab', AdaBoostClassifier(
                            base_estimator=DecisionTreeClassifier(max_depth=1,
                                                                  class_weight="balanced"),
                            learning_rate=0.3,
                            n_estimators=2,
                            algorithm="SAMME.R",
                            random_state=1234))])


# combine everything in one voting classifier
vote_clf = VotingClassifier(estimators=[('lr', lr_pipe), ('rf', rf_pipe), ('ab', ab_pipe)],
                            voting="soft", n_jobs=-1)
_ = vote_clf.fit(X=X_layer1,y=y_layer1)
\`\`\`


\`\`\`python
# let's do a prediction for the layer 2 test set
y_pred = vote_clf.predict(X_layer2)
print("Voting classifier on layer 2 test set.\\nConfusion matrix:")
print(confusion_matrix(y_layer2, y_pred))
print("Recall: {:0.3f}".format(recall_score(y_layer2, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_layer2, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_layer2, y_pred)))
\`\`\`

    Voting classifier on layer 2 test set.
    Confusion matrix:
    [[3166   86]
     [  28  300]]
    Recall: 0.915
    Precision: 0.777
    Accuracy: 0.968
    

As we can see, combining the classifiers into a voting ensemble did not improv the performance. Let's have a look at which examples are being missclassifed.


\`\`\`python
# train all the classifierers and get the false negatives from the prediction
lr_pipe.fit(X_layer1,y_layer1)
lr_y_pred = lr_pipe.predict(X_layer2)
lr_fn = [ i for i,x in enumerate(zip(y_layer2,lr_y_pred)) if x[0]!=x[1] and x[0]==1]

rf_pipe.fit(X_layer1,y_layer1)
rf_y_pred = rf_pipe.predict(X_layer2)
rf_fn = [ i for i,x in enumerate(zip(y_layer2,rf_y_pred)) if x[0]!=x[1] and x[0]==1]

ab_pipe.fit(X_layer1,y_layer1)
ab_y_pred = ab_pipe.predict(X_layer2)
ab_fn = [ i for i,x in enumerate(zip(y_layer2,rf_y_pred)) if x[0]!=x[1] and x[0]==1]

vote_clf.fit(X_layer1,y_layer1)
vote_y_pred = vote_clf.predict(X_layer2)
vote_fn = [ i for i,x in enumerate(zip(y_layer2,rf_y_pred)) if x[0]!=x[1] and x[0]==1]

# count how often each false negative got missclassified
print(pd.Series(lr_fn+ab_fn+rf_fn+vote_fn).value_counts())
\`\`\`

    2805    4
    2960    4
    2062    4
    2333    4
    554     4
    1581    4
    1590    4
    2879    4
    1603    4
    326     4
    3416    4
    122     4
    2234    4
    134     4
    244     4
    140     4
    2548    4
    1222    4
    236     4
    230     4
    3342    4
    3061    4
    3019    4
    3530    4
    1793    4
    186     4
    432     4
    686     4
    2141    3
    3291    3
    2225    3
    2784    3
    795     3
    1846    1
    2673    1
    383     1
    596     1
    dtype: int64
    

Looks like almost all false negatives got missclassified by all classifiers.

We will now try to blend the output of all these classifiers into another classifer. Maybe there's a pattern in why all these false negatives get missclassified and maybe a second layer classifier will be abel to pick up this pattern and correct it.

We will use our layer 2 set to train the bleder layer but first, we'll have to send it through layer 1 (which was trained on the layer 1 set) and predict each instances class probability, which will be used as feature for layer 2. Note that when  we are ust dealing with the probability for a class, we cannot optimize for recall because one can simply categorize every instance as a Pulsar - then there would be no false negatives but a loooot of false positives and the whole task wouldn't make any sense. Therefore, we will use the F1 score, which is a mix of recall and precision. It is better suited than accuracy for strong class imbalances like in this case.


\`\`\`python
# we'll predict probability for the positive class using all calssifiers
def blender_features(X_layer2):
    lr_y_prob = lr_pipe.predict_proba(X_layer2)[:,1]
    rf_y_prob = rf_pipe.predict_proba(X_layer2)[:,1]
    ab_y_prob = ab_pipe.predict_proba(X_layer2)[:,1]
    vote_y_prob = vote_clf.predict_proba(X_layer2)[:,1]
# and these probabilities will be the new features for our second blender layer
    return np.c_[lr_y_prob,rf_y_prob,ab_y_prob,vote_y_prob]
\`\`\`


\`\`\`python
# do a grid search on a logistic regression classifier which serves as the blender
X_blend = blender_features(X_layer2)

param_grid = {"C":[1e-10,1e-5,1e-5,1e-2,1]}
blender_grid = GridSearchCV(LogisticRegression(class_weight="balanced",
                                         solver="lbfgs",
                                         random_state=1234),
                            param_grid=param_grid, scoring="f1",
                            iid=False, cv=5,n_jobs=-1)

blender_grid.fit(X_blend,y_layer2)
print("Best parameter (CV F1 score=%0.3f):" % blender_grid.best_score_)
print(blender_grid.best_params_)
\`\`\`

    Best parameter (CV F1 score=0.839):
    {'C': 1e-05}
    


\`\`\`python
# let's do a the final test prediction for the actual test set
X_blend_test = blender_features(X_test)
y_pred = blender_grid.predict(X_blend_test)
print("Blender classifier on final test set.\\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Recall: {:0.3f}".format(recall_score(y_test, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_test, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_test, y_pred)))
\`\`\`

    Blender classifier on final test set.
    Confusion matrix:
    [[3167   85]
     [  30  298]]
    Recall: 0.909
    Precision: 0.778
    Accuracy: 0.968
    

Okay, that's a little disappointing. Not much of an improvement here. Looks like all the fancy ensembles we tried are not performing much better than the first simple logistic regression we used as a benchmark. 

That probably means that we do not just deal with random noise that may cause trouble for some types of classifiers, and which we can try to cancel out by using ensembles. All classifiers seemed to misclassify the same instances. Probably there is something systematically different about the misclassified false negative Pulsars, something that makes them look more like the spurious signals in the data set. 

If we really wanted to solve this problem we should investigate these examples in detail now, find out what is special about them, and engineer this into features which a classification algorithm can work with in a better way. But hey, this was just supposed to be an example for demonstrating ensemble methods; hence, let's call it a day and end here.

### Dimensionality reduction on colonoscopy video data <a id="colonoscopy"></a>

Machine learning is becoming more successful in many medical applications, especially in analyzing image or video data. The field of proctology is no exception. I find this example particularly interesting as one of my best friends has to get regular colonoscopies as a measure of colon cancer prophylactics and I'd like to tell her if machine learning can assist the doctors in classifying whatever they find in her behind as benign or malignant. You can find the Jupyter Notebook and data [here](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository/tree/master/Dimensionality-reduction-for-colonoscopy-data).

This data set contains 76 instances of gastrointestinal lesions, two malignant and one benign type, from regular colonoscopy data. It was contributed to the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Gastrointestinal+Lesions+in+Regular+Colonoscopy) by Pablo Mesejo and Daniel Pizarro. You can find a detailed description on the website. They also provided a summary of how human experts performed on classifying the examples.

#### Frame the Problem
In this example, we want to classify each lesions as either serrated adenomas, adenoma, or hyperplastic lesion. The first two are considered malignant, the latter as benign. Hence, we can also treat it as a binary classification problem (the authors of the data set labeled this binary case as *resection* vs. *no-resection*). 

What is special about this data set is that we are dealing with a lot of features but only a few instances. From each video, there are 422 2D textural features, 76 2D color features, 200 3D shape features, and each of those recorded under two different light conditions. A total of 1397 features and only 76 instances. Most classification algorithms will have problems with this ratio. Support vector machines are known for being able to deal with many features given few instances, but this ratio might be too extreme even for them. So what can we do about it? 

In this example we will try to apply principal component analysis (PCA) to reduce the dimensionality of the feature space while still preserving much information of the original data. PCA will project the original high-dimensional data on a lower dimensional hyperplane (which is spanned by the principal components). We can either specify a number of dimensions and pick the PCs that preserve the most variance or we specify ho much variance should be preserved and pick the number of PCs accordingly. 

I would usually chose recall as optimization metric because I think it is important to not miss potential cancer; but since we want to compare to human performance (who probably want to get all classifications right) and we have a class imbalance, I'll choose the F1 score.


\`\`\`python
# import libraries 
import numpy as np # numerical computation
import pandas as pd # data handling
import warnings
warnings.filterwarnings('ignore')
# visulalization
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("darkgrid")
%matplotlib notebook
\`\`\`

#### Data preparation
Let's import the data into data frame hand have a look.


\`\`\`python
colon = pd.read_csv("data.txt")
colon.head()
\`\`\`




<div class="post-page-table">
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
      <th>adenoma_1</th>
      <th>adenoma_1.1</th>
      <th>adenoma_8</th>
      <th>adenoma_8.1</th>
      <th>adenoma_9</th>
      <th>adenoma_9.1</th>
      <th>adenoma_10</th>
      <th>adenoma_10.1</th>
      <th>adenoma_11</th>
      <th>adenoma_11.1</th>
      <th>...</th>
      <th>serrated_5</th>
      <th>serrated_5.1</th>
      <th>serrated_6</th>
      <th>serrated_6.1</th>
      <th>serrated_7</th>
      <th>serrated_7.1</th>
      <th>serrated_8</th>
      <th>serrated_8.1</th>
      <th>serrated_9</th>
      <th>serrated_9.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138.120000</td>
      <td>127.990000</td>
      <td>80.415000</td>
      <td>90.896000</td>
      <td>106.160000</td>
      <td>147.090000</td>
      <td>148.730000</td>
      <td>126.050000</td>
      <td>109.130000</td>
      <td>129.700000</td>
      <td>...</td>
      <td>114.590000</td>
      <td>86.424000</td>
      <td>163.680000</td>
      <td>71.638000</td>
      <td>180.110000</td>
      <td>136.550000</td>
      <td>96.852000</td>
      <td>157.810000</td>
      <td>93.569000</td>
      <td>95.543000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1606.800000</td>
      <td>3377.900000</td>
      <td>1852.100000</td>
      <td>1904.300000</td>
      <td>1184.400000</td>
      <td>822.320000</td>
      <td>2412.500000</td>
      <td>4752.200000</td>
      <td>999.390000</td>
      <td>599.950000</td>
      <td>...</td>
      <td>3014.500000</td>
      <td>3500.900000</td>
      <td>3253.100000</td>
      <td>1822.200000</td>
      <td>1198.500000</td>
      <td>1316.300000</td>
      <td>2071.300000</td>
      <td>2732.300000</td>
      <td>1163.600000</td>
      <td>2240.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.003875</td>
      <td>0.003564</td>
      <td>0.004761</td>
      <td>0.004147</td>
      <td>0.005518</td>
      <td>0.003871</td>
      <td>0.003336</td>
      <td>0.004188</td>
      <td>0.005541</td>
      <td>0.005917</td>
      <td>...</td>
      <td>0.004444</td>
      <td>0.003409</td>
      <td>0.004869</td>
      <td>0.004148</td>
      <td>0.003273</td>
      <td>0.002442</td>
      <td>0.004379</td>
      <td>0.004015</td>
      <td>0.002199</td>
      <td>0.004803</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 152 columns</p>
</div>



In this raw format, each column represents an instance measured at one of the two different light conditions, and the rows represents a features. The first row is the class label and the second row the light condition (1 or 2). We'll first have to merge the two light conditions per instance and than transpose the data frame so that we have the data in a [tidy format](https://vita.had.co.nz/papers/tidy-data.pdf#targetText=This%20paper%20tackles%20a%20small,observational%20unit%20is%20a%20table.).


\`\`\`python
# get the column names for light conditions 1 and 2
light1cols = colon.loc[1,colon.iloc[1]==1].index
light2cols = colon.loc[1,colon.iloc[1]==2].index
# create seperate data frames for the two conditions
light1 = colon[light1cols]
light2 = colon[light2cols]
# give them the same column names so that they can be appended 
light2.columns = light1.columns
# append data frame while dropping the light condtion and class label from one of them
colon = light1.append(light2.iloc[2:])
# drop the light condition from the other one
colon = colon.drop(1,axis=0).reset_index(drop=True)
# transpose the data frame so that instances are rows and features are columns
colon = colon.T.reset_index(drop=True).rename(columns={0:"label"})
\`\`\`


\`\`\`python
colon.head()
\`\`\`




<div class="post-page-table">
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
      <th>label</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1387</th>
      <th>1388</th>
      <th>1389</th>
      <th>1390</th>
      <th>1391</th>
      <th>1392</th>
      <th>1393</th>
      <th>1394</th>
      <th>1395</th>
      <th>1396</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>138.120</td>
      <td>1606.80</td>
      <td>0.003875</td>
      <td>0.005880</td>
      <td>0.005213</td>
      <td>0.006935</td>
      <td>0.007333</td>
      <td>0.009580</td>
      <td>0.007380</td>
      <td>...</td>
      <td>0.013994</td>
      <td>0.013532</td>
      <td>0.013157</td>
      <td>0.012743</td>
      <td>0.012613</td>
      <td>0.012422</td>
      <td>0.012252</td>
      <td>0.011377</td>
      <td>0.011198</td>
      <td>0.011131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>90.896</td>
      <td>1904.30</td>
      <td>0.004147</td>
      <td>0.006728</td>
      <td>0.005061</td>
      <td>0.006879</td>
      <td>0.007948</td>
      <td>0.009525</td>
      <td>0.010492</td>
      <td>...</td>
      <td>0.003564</td>
      <td>0.003380</td>
      <td>0.003232</td>
      <td>0.003200</td>
      <td>0.003006</td>
      <td>0.002985</td>
      <td>0.002922</td>
      <td>0.002631</td>
      <td>0.002610</td>
      <td>0.002531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>147.090</td>
      <td>822.32</td>
      <td>0.003871</td>
      <td>0.005211</td>
      <td>0.005834</td>
      <td>0.006971</td>
      <td>0.011036</td>
      <td>0.012802</td>
      <td>0.011083</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>148.730</td>
      <td>2412.50</td>
      <td>0.003336</td>
      <td>0.007695</td>
      <td>0.004139</td>
      <td>0.005736</td>
      <td>0.005794</td>
      <td>0.006697</td>
      <td>0.007721</td>
      <td>...</td>
      <td>0.020822</td>
      <td>0.020115</td>
      <td>0.019595</td>
      <td>0.019252</td>
      <td>0.018897</td>
      <td>0.018177</td>
      <td>0.018158</td>
      <td>0.017587</td>
      <td>0.017109</td>
      <td>0.016648</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>129.700</td>
      <td>599.95</td>
      <td>0.005917</td>
      <td>0.007934</td>
      <td>0.006976</td>
      <td>0.007695</td>
      <td>0.008404</td>
      <td>0.008825</td>
      <td>0.010306</td>
      <td>...</td>
      <td>0.000140</td>
      <td>0.000129</td>
      <td>0.000117</td>
      <td>0.000111</td>
      <td>0.000106</td>
      <td>0.000098</td>
      <td>0.000093</td>
      <td>0.000082</td>
      <td>0.000079</td>
      <td>0.000076</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1397 columns</p>
</div>



Now that looks a lot tidier. Let's also change the class labels: 1 becomes 0 for benign; 2 and 3 become 1 for malignant. 


\`\`\`python
colon.label[colon.label==1] = 0
colon.label[(colon.label==2) | (colon.label==3)] = 1
colon.sample(5)
\`\`\`




<div class="post-page-table">
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
      <th>label</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1387</th>
      <th>1388</th>
      <th>1389</th>
      <th>1390</th>
      <th>1391</th>
      <th>1392</th>
      <th>1393</th>
      <th>1394</th>
      <th>1395</th>
      <th>1396</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>127.47</td>
      <td>1406.80</td>
      <td>0.002802</td>
      <td>0.004276</td>
      <td>0.003806</td>
      <td>0.005146</td>
      <td>0.005410</td>
      <td>0.006105</td>
      <td>0.007818</td>
      <td>...</td>
      <td>0.003110</td>
      <td>0.003053</td>
      <td>0.002898</td>
      <td>0.002774</td>
      <td>0.002632</td>
      <td>0.002493</td>
      <td>0.002450</td>
      <td>0.002312</td>
      <td>0.002179</td>
      <td>0.002090</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.0</td>
      <td>140.73</td>
      <td>711.59</td>
      <td>0.006186</td>
      <td>0.009827</td>
      <td>0.006869</td>
      <td>0.009844</td>
      <td>0.010494</td>
      <td>0.012854</td>
      <td>0.012929</td>
      <td>...</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1.0</td>
      <td>163.11</td>
      <td>1385.90</td>
      <td>0.008264</td>
      <td>0.011529</td>
      <td>0.006550</td>
      <td>0.007291</td>
      <td>0.012066</td>
      <td>0.013678</td>
      <td>0.015943</td>
      <td>...</td>
      <td>0.000050</td>
      <td>0.000048</td>
      <td>0.000046</td>
      <td>0.000046</td>
      <td>0.000044</td>
      <td>0.000039</td>
      <td>0.000033</td>
      <td>0.000032</td>
      <td>0.000030</td>
      <td>0.000027</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.0</td>
      <td>154.63</td>
      <td>566.70</td>
      <td>0.005386</td>
      <td>0.008225</td>
      <td>0.006184</td>
      <td>0.008711</td>
      <td>0.006867</td>
      <td>0.008170</td>
      <td>0.013066</td>
      <td>...</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>158.80</td>
      <td>3795.40</td>
      <td>0.005593</td>
      <td>0.010721</td>
      <td>0.009279</td>
      <td>0.011931</td>
      <td>0.014284</td>
      <td>0.013749</td>
      <td>0.018370</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1397 columns</p>
</div>




\`\`\`python
# look for missing values
colon.isna().sum()[colon.isna().sum()>0]
\`\`\`




    Series([], dtype: int64)




\`\`\`python
# look at class distribution
colon.label.value_counts()
\`\`\`




    1.0    55
    0.0    21
    Name: label, dtype: int64



Looks like we don't have to clean the data anymore. We should now split off our test set.


\`\`\`python
from sklearn.model_selection import StratifiedShuffleSplit
# put 20%  of the data aside as a test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split.split(colon, colon["label"]):
    colon_train = colon.loc[train_index]
    colon_test = colon.loc[test_index]
    X_train = colon_train.iloc[:,1:]
    y_train = colon_train.loc[:,"label"]
    X_test = colon_test.iloc[:,1:]
    y_test = colon_test.loc[:,"label"]

\`\`\`

#### Selecting and tuning models
Let's try to get a base line first by training an SVC model with all the features and no preprocessing besides standard scaling. We can than compare the performance of the lower dimensionality models with this benchmark.


\`\`\`python
# import all sklearn functions needed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# put a standard scaler and a support vector calssifier in a pipeline
pipe = Pipeline(steps=[
               ("scaler",StandardScaler()),
               ("svc",SVC(class_weight="balanced"))])

# define paramter space to search durring grid search
param_grid = {"svc__kernel":["linear","rbf"],
              "svc__C":np.logspace(-6,4,5),
              "svc__gamma":np.logspace(-8,1,5)}

# perform grid search
grid = GridSearchCV(pipe, param_grid=param_grid,
                    n_jobs=-1, cv=6, scoring="f1",
                    iid=False, verbose=0)
grid.fit(X_train,y_train)

print("Cross validation Grid search:")
print("Best parameter (CV F1 score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

print("\\nPerformance on the test set:\\n")
print("Confusion matrix:")
y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("F1: {:0.3f}".format(f1_score(y_test, y_pred)))
print("Recall: {:0.3f}".format(recall_score(y_test, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_test, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_test, y_pred)))
\`\`\`

    Cross validation Grid search:
    Best parameter (CV F1 score=0.880):
    {'svc__C': 0.1, 'svc__gamma': 1e-08, 'svc__kernel': 'linear'}
    
    Performance on the test set:
    
    Confusion matrix:
    [[ 4  0]
     [ 2 10]]
    F1: 0.909
    Recall: 0.833
    Precision: 1.000
    Accuracy: 0.875
    

We actually perform already very well without any fancy preprocessing, already beating the human experts who's binary classification accuracy is around 0.796. But we still have a few false negatives in our test set classification which can be a severe problem for malignant tumors. If a patient is told that they're fine, the tumor may get a lot worse until they notice that the diagnosis was wrong. Maybe we can improve our benchmark results.

Let's start with a simple PCA with only two PCs so that we can plot our data points 


\`\`\`python
from sklearn.decomposition import PCA
# do a PCA with 2 PCs
# Note: the infput data is centred aroud 0 automatically before the PCA,
# the whiten parameter scales the output to unit variance, so it basically acts
# like a standard scaler
pca = PCA(n_components=2, whiten=True)
X_train_pca = pca.fit_transform(X_train)
print("Variance explained by the first two PCs: {:0.3f}".format(sum(pca.explained_variance_ratio_)))
\`\`\`

    Variance explained by the first two PCs: 0.921
    

Wow! 92% of the original training data is preserved when projecting it from 1396 dimensions to only two dimensions. Lets plot it: 


\`\`\`python
# split instances into malignant and benign
idx_b = np.argwhere(y_train==0).reshape(1,-1)
idx_m = np.argwhere(y_train==1).reshape(1,-1)
malignant = X_train_pca[idx_m].reshape(-1,2)
benign = X_train_pca[idx_b].reshape(-1,2)

# plot on a scatter plot
plt.plot(malignant[:,0],malignant[:,1],"^r",label="malignant",ms=8,alpha=.5)
plt.plot(benign[:,0],benign[:,1],".b",label="benign",ms=10,alpha=.5)
_ = plt.legend()
\`\`\`


<img style="width: 50%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/colonoscopy_output_19_0.png">


We can see that benign and malignant lesions tend to different regions of the figure but there i some significant overlap of points which would probably hard to deconvolve. Maybe we've lost too much of the original data's information and we should chose the number of PCs according to how much variance we want to preserve. We can specify that percentage instead the number of PCs.


\`\`\`python
# do a PCA preserving 0.95 of the original data's variance
var = 0.95
pca = PCA(n_components=var)
X_train_pca = pca.fit_transform(X_train)
n_comp = len(pca.components_)
print("Preserving {} variance requires {} PCs.".format(var,n_comp))
\`\`\`

    Preserving 0.95 variance requires 3 PCs.
    


\`\`\`python
# do a PCA preserving 0.99 of the original data's variance
var = 0.99
pca = PCA(n_components=var)
X_train_pca = pca.fit_transform(X_train)
n_comp = len(pca.components_)
print("Preserving {} variance requires {} PCs.".format(var,n_comp))
\`\`\`

    Preserving 0.99 variance requires 6 PCs.
    


\`\`\`python
# do a PCA preserving 0.999 of the original data's variance
var = 0.999
pca = PCA(n_components=var)
X_train_pca = pca.fit_transform(X_train)
n_comp = len(pca.components_)
print("Preserving {} variance requires {} PCs.".format(var,n_comp))
\`\`\`

    Preserving 0.999 variance requires 11 PCs.
    

The amount of preserved variance can be used as a hyper-parameter in a grid search.


\`\`\`python
# put a standard-scaled PCA and a support vector calssifier in a pipeline
pipe = Pipeline(steps=[
               ("pca",PCA(whiten=True)),
               ("svc",SVC(kernel="linear",class_weight="balanced"))])

# define paramter space to search durring grid search
param_grid = {"svc__C":np.logspace(-4,6,5),
              "pca__n_components":[0.95,0.99,0.999]}

# perform grid search
grid = GridSearchCV(pipe, param_grid=param_grid,
                    n_jobs=-1, cv=6, scoring="f1",
                    iid=False, verbose=0)
grid.fit(X_train,y_train)

print("Cross validation Grid search:")
print("Best parameter (CV F1 score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

print("\\nPerformance on the test set:\\n")
print("Confusion matrix:")
y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("F1: {:0.3f}".format(f1_score(y_test, y_pred)))
print("Recall: {:0.3f}".format(recall_score(y_test, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_test, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_test, y_pred)))
\`\`\`

    Cross validation Grid search:
    Best parameter (CV F1 score=0.916):
    {'pca__n_components': 0.999, 'svc__C': 3162.2776601683795}
    
    Performance on the test set:
    
    Confusion matrix:
    [[ 4  0]
     [ 2 10]]
    F1: 0.909
    Recall: 0.833
    Precision: 1.000
    Accuracy: 0.875
    

The cross-validation score was slightly improved, but in our test set there are still two instances misclassified. There is a version of kernel PCA (which basically employs the kernel trick like support vector machines) which allows it to perform nonlinear projections for dimensionality reduction. Let's see if we can improve our results with kernel PCA. Unfortunately we can not directly specify the amount of variance we want to preserve, so we'll have to calculate manually how many components will be needed for a certain desired preserved variance given a certain kernel (e.g. linear, rbf, polynomial, cosine, using default parameters).


\`\`\`python
from sklearn.decomposition import KernelPCA

# set up a kPCA with linear kernel and calculate all PCs
kpca = KernelPCA(kernel="linear")
X_train_kpca = kpca.fit_transform(X_train)
# calculate the explained variance
explained_variance = np.var(X_train_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumsum = np.cumsum(explained_variance_ratio)
# print results
print("For kPCA with linear kernel:")
for v in [0.95,0.99,0.999]:
    for i,c in enumerate(cumsum):
        if c>v:
            print("Preserving {} variance requires {} PCs.".format(v,i+1))
            break
\`\`\`

    For kPCA with linear kernel:
    Preserving 0.95 variance requires 3 PCs.
    Preserving 0.99 variance requires 6 PCs.
    Preserving 0.999 variance requires 11 PCs.
    


\`\`\`python
# set up a kPCA with rbf kernel and calculate all PCs
kpca = KernelPCA(kernel="rbf")
X_train_kpca = kpca.fit_transform(X_train)
# calculate the explained variance
explained_variance = np.var(X_train_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumsum = np.cumsum(explained_variance_ratio)
# print results
print("For kPCA with rbf kernel:")
for v in [0.95,0.99,0.999]:
    for i,c in enumerate(cumsum):
        if c>v:
            print("Preserving {} variance requires {} PCs.".format(v,i+1))
            break
\`\`\`

    For kPCA with rbf kernel:
    Preserving 0.95 variance requires 57 PCs.
    Preserving 0.99 variance requires 59 PCs.
    Preserving 0.999 variance requires 59 PCs.
    


\`\`\`python
# set up a kPCA with polynomial kernel and calculate all PCs
kpca = KernelPCA(kernel="poly")
X_train_kpca = kpca.fit_transform(X_train)
# calculate the explained variance
explained_variance = np.var(X_train_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumsum = np.cumsum(explained_variance_ratio)
# print results
print("For kPCA with polynomial kernel:")
for v in [0.95,0.99,0.999]:
    for i,c in enumerate(cumsum):
        if c>v:
            print("Preserving {} variance requires {} PCs.".format(v,i+1))
            break
\`\`\`

    For kPCA with polynomial kernel:
    Preserving 0.95 variance requires 1 PCs.
    Preserving 0.99 variance requires 2 PCs.
    Preserving 0.999 variance requires 4 PCs.
    


\`\`\`python
# set up a kPCA with cosine kernel and calculate all PCs
kpca = KernelPCA(kernel="cosine")
X_train_kpca = kpca.fit_transform(X_train)
# calculate the explained variance
explained_variance = np.var(X_train_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumsum = np.cumsum(explained_variance_ratio)
# print results
print("For kPCA with cosine kernel:")
for v in [0.95,0.99,0.999]:
    for i,c in enumerate(cumsum):
        if c>v:
            print("Preserving {} variance requires {} PCs.".format(v,i+1))
            break
\`\`\`

    For kPCA with cosine kernel:
    Preserving 0.95 variance requires 8 PCs.
    Preserving 0.99 variance requires 12 PCs.
    Preserving 0.999 variance requires 18 PCs.
    

If we want to use both the kernel type and the preserved variance level in hyper-parameter tuning, we should write our own transformer class which handles that easily.


\`\`\`python
from sklearn.base import BaseEstimator, TransformerMixin

class kPCA(BaseEstimator, TransformerMixin):
    """This class allows to tune kernel-type and variance-to-preserve as hyper-paramters"""
    
    pc_dict = {"linear":{0.95:3,0.99:6,0.999:11},
               "rbf":{0.95:57,0.99:59,0.999:59},
               "poly":{0.95:1,0.99:2,0.999:4},
               "cosine":{0.95:8,0.99:12,0.999:18}}
    
    def __init__(self, kernel="linear", var=0.95):
        self.kernel = kernel
        self.var = var
        self.kpca = KernelPCA(kernel=self.kernel,
                              n_components=self.pc_dict[kernel][var])
        
    def fit(self, X, y=None):
        self.kpca.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.kpca.transform(X)
\`\`\`

Let's see if we can up our performance in a final grid search with different kernel PCAs.


\`\`\`python
# put the custom kPCA, a standard scaler, and a support vector calssifier in a pipeline
pipe = Pipeline(steps=[
               ("kpca",kPCA()),
               ("scaler", StandardScaler()),
               ("svc",SVC(kernel="linear",class_weight="balanced"))])

# define paramter space to search durring grid search
param_grid = {"kpca__kernel":["linear","rbf","poly","cosine"],
              "kpca__var":[0.95,0.99,0.999],
              "svc__C":np.logspace(-6,6,4)}

# perform grid search
grid = GridSearchCV(pipe, param_grid=param_grid,
                    n_jobs=-1, cv=6, scoring="f1",
                    iid=False, verbose=0)
grid.fit(X_train,y_train)

print("Cross validation Grid search:")
print("Best parameter (CV F1 score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

print("\\nPerformance on the test set:\\n")
print("Confusion matrix:")
y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("F1: {:0.3f}".format(f1_score(y_test, y_pred)))
print("Recall: {:0.3f}".format(recall_score(y_test, y_pred)))
print("Precision: {:0.3f}".format(precision_score(y_test, y_pred)))
print("Accuracy: {:0.3f}".format(accuracy_score(y_test, y_pred)))
\`\`\`

    Cross validation Grid search:
    Best parameter (CV F1 score=0.870):
    {'kpca__kernel': 'linear', 'kpca__var': 0.95, 'svc__C': 100.0}
    
    Performance on the test set:
    
    Confusion matrix:
    [[ 4  0]
     [ 2 10]]
    F1: 0.909
    Recall: 0.833
    Precision: 1.000
    Accuracy: 0.875
    

Well, looks like we weren't really able to achieve much of an improvement by dimensionality reduction but we also didn't make it worse. That means that the information that is in the data was as easy to grasp for the algorithm in higher dimensional space as in dimensionality reduced space. We do gain one big advantage through dimensionality reduction though: Significantly less computational cost! For this example with only 76 instances this may not seem relevant, but for much larger data sets it will be a massive speed-up if we can reduce the number of features from hundreds to only about a dozen without sacrificing too much classification accuracy (or what ever score we use).

### Optimized deep neural networks learning poker hands <a id="poker"></a>

This data set contains examples of possible poker hands which can be used to let a machine learning algorithm learn the rules of the game. It was used in a [research paper](https://pdfs.semanticscholar.org/c068/ea7807367573f4b5f98c0681fca665e9ef74.pdf) in which the authors, R. Cattral, F. Oppacher, and D. Deugo, used evolutionary and symbolic machine learning methods to extract comprehensible and strong rules (the rules of poker in this case) from it. We will try to achieve this with an optimized neural network. You can find the Jupyter Notebook and the data [here](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository/tree/master/Deep-learning-on-poker-hands).

The data set was contributed to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Poker+Hand) by Robert Cattral and Franz Oppacher from the Department of Computer Science of Carleton University, which is where I got it from.

#### Frame the Problem
In this small project, we're not going all the way to building a full poker-bot that can play the entire game of poker against other agents. We just want to see if and how well an algorithm can understand the rules of how poker hands (the combination of cards that result in a certain score) are composed.

The data we will consider contains one million instances of poker hands. Each poker hand in the data set is an example consisting of five cards which are drawn from a poker deck of 52 cards. Each card has two attributes, its suit (S, 1 to 4, representing Hearts, Spades, Diamonds, and Clubs) and its rank (C, 1 to 13, representing Ace, 2, 3, ... , Queen, and King). For each instance (poker hand) that results in 10 features. There are ten possible classes (0 to 9) which correspond to the card [combinations](https://en.wikipedia.org/wiki/Poker#Gameplay) that can be observed in the game of poker: nothing in hand, one pair, two pairs, three of a kind, straight, flush, full house, four of a kind, straight flush, and royal flush.

We will treat this as a supervised multi-class classification problem. The rules of this classification are fairly complicated but we also have a lot of instances to train on, which makes the problem very suitable for deep neural networks. We will work with a multilayer perceptron architecture, implement it with \`TensorFlow 2.0\`'s \`Keras\` API, and optimize its hyper-parameters with the \`hyperas\` library using a bayesian optimization.


\`\`\`python
# import the libraries we'll need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("darkgrid")
%matplotlib notebook
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras as keras
\`\`\`

#### Data preparation
Let's import the data into data frame hand have a look.


\`\`\`python
# import data from CSV file
df = pd.read_csv("poker-hand-testing.data",
                 names = ["S1","C1","S2","C2","S3","C3",
                          "S4","C4","S5","C5","target"])

# order the columns to have the ranks (C, numerical) 
# and suits (S, categorical) together
df = df[["C1","C2","C3","C4","C5",
         "S1","S2","S3","S4","S5",
         "target"]]

# let's have a look by taking a random sample
df.sample(10)
\`\`\`




<div class="post-page-table">
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
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>808385</th>
      <td>3</td>
      <td>8</td>
      <td>6</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>234800</th>
      <td>9</td>
      <td>12</td>
      <td>5</td>
      <td>10</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>196245</th>
      <td>12</td>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>9</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>609543</th>
      <td>12</td>
      <td>11</td>
      <td>13</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>961547</th>
      <td>11</td>
      <td>3</td>
      <td>11</td>
      <td>10</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>325426</th>
      <td>9</td>
      <td>12</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>752949</th>
      <td>13</td>
      <td>7</td>
      <td>4</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>958119</th>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>13</td>
      <td>10</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116907</th>
      <td>1</td>
      <td>10</td>
      <td>8</td>
      <td>4</td>
      <td>11</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>439993</th>
      <td>2</td>
      <td>9</td>
      <td>4</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




\`\`\`python
# print the data frame's info
df.info()
\`\`\`

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 11 columns):
    C1        1000000 non-null int64
    C2        1000000 non-null int64
    C3        1000000 non-null int64
    C4        1000000 non-null int64
    C5        1000000 non-null int64
    S1        1000000 non-null int64
    S2        1000000 non-null int64
    S3        1000000 non-null int64
    S4        1000000 non-null int64
    S5        1000000 non-null int64
    target    1000000 non-null int64
    dtypes: int64(11)
    memory usage: 83.9 MB
    


\`\`\`python
# look for missing values
df.isna().sum()
\`\`\`




    C1        0
    C2        0
    C3        0
    C4        0
    C5        0
    S1        0
    S2        0
    S3        0
    S4        0
    S5        0
    target    0
    dtype: int64



We can see that there are apparently no missing values. Let's look at how the target classes are distributed.


\`\`\`python
df.target.value_counts().sort_index()
\`\`\`




    0    501209
    1    422498
    2     47622
    3     21121
    4      3885
    5      1996
    6      1424
    7       230
    8        12
    9         3
    Name: target, dtype: int64



As we can see, we're dealing with a very strong class imbalance. Some of the poker hands are much rarer than others. We can try to account for this by handing class weights to the classifier but it will certainly be very difficult, if not impossible, to correctly classify e.g. a straight flush or royal flush.

Let's also have a look at how the features are distributed to make sure there are no outliers.


\`\`\`python
# plot feature distribution as histograms
_ = df.hist(bins=13,figsize=(6,5))
plt.tight_layout()
\`\`\`


<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/poker_output_10_0.png">


Great, looks like the data is actually distributed as specified. No need to account for invalid values. We can now proceed to splitting of test and validation data sets. Because of the class imbalance, we will stratify the split so that approximately equal class distributions will be present in all sets. For the very rare classes this may not work perfectly. We'll also compute the class weights already.


\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# split off a test set and stratify for classes (target)
train_full, test = train_test_split(df,
                                    stratify = df["target"],
                                    test_size=0.2, 
                                    random_state=123)

# split into a training and a validation set and stratify for classes (target)
train, valid = train_test_split(train_full,
                                stratify = train_full["target"],
                                test_size=0.2,
                                random_state=123)

# compute class weigths which will be used to account for class imbalance
class_weights = compute_class_weight("balanced",
                                     np.unique(train["target"]),
                                     train["target"])
\`\`\`

I really like using handy \`pandas\` and \`sklearn\` functions for exploring and preprocessing moderate-sized data. But for really big data that has to be distributed over several machines, we might prefer to use \`dask\` data frames or to do everything just in \`TensorFlow\`. \`TensorFlow\`'s data API can read data from different sources, handle preprocessing (even though I personally don't always find it very handy), and provide tensor datasets and \`TensorFlow\` also offers different kinds of feature columns which can be directly fed into a model. We'll use these features here to explore them a bit, even though we could also just use \`sklearn\`'s preprocessing on \`pandas\`'s data frames here on my single poor old laptop.

We'll first define a function that turns the \`pandas\` data frames into \`TensorFlow\` data sets.


\`\`\`python
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """
    A function that turns pandas data frames into TensorFlow data sets.
    Params:
    ----------
    dataframe: input pandas data frame
    shuffle: if True, the data set will be pre-shuffled
    batch_size: batch size, meaning number of instances that 
                will be passed to model per optimization step
    Returns:
    ---------
    ds: the output data set
    """
    # get copy of data frame
    dataframe = dataframe.copy()
    
    # extract class labels
    labels = dataframe["target"]
    
    # make data set from features (first 10 columns) and labels
    # note: it's important to convert data type from int to float32
    # here, TF doesn't do that automatically and will throw error
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe.iloc[:,:10]
                                                  .astype(np.float32)), 
                                             labels))
    # shuffle if desired
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    
    # enable infinite repeat with given batch size and prefetch
    ds = ds.repeat().batch(batch_size).prefetch(1)
    
    return ds
\`\`\`

Now we can actually transform the data frames into \`TensofFlow\` data sets which will be fed to our model. But we'll have to choose the batch size first, how many instances will be passed through the model per optimization step.

The batch size should be large enough to give a precise enough estimate of gradients during optimization but not so large that it significantly slows down the training iterations. In practice 32 is often chosen as default. At least that's what I understood from \`TensofFlow\` tutorials and a couple of deep learning books. I have no reason not to trust this advice here.


\`\`\`python
batch_size = 32
# create tensorflow data sets from data frames
train_ds = df_to_dataset(train, batch_size=batch_size)
valid_ds = df_to_dataset(valid, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
\`\`\`

We will want to normalize the numerical data (first 5 columns) before we send it into the model. We can provide a normalization function to the numerical feature column later but we'll first have to compute each feature's mean and standard deviation. We can compute them as \`TensorFlow\` tensors as well.


\`\`\`python
# calculate means and standard deviations for the numerical features
X_means = tf.math.reduce_mean(train.iloc[:,:5].values.astype(np.float32),axis=0)
X_stds = tf.math.reduce_std(train.iloc[:,:5].values.astype(np.float32),axis=0)
# since they all have approximately the same distribution, the means should all
# be around 7.0 and the standard deviations around 3.74
\`\`\`

We can now define the feature columns. We will have 5 numerical columns corresponding to the cards' ranks, which we will normalize. There will also be 5 categorical columns for the suits. Since they are ordinal and represented as integers here, we can first put them in 5 numerical columns and then transform these into 5 bucketized columns with 4 buckets each, one for each suit.


\`\`\`python
# collect all feature columns in a list which we'll
# later feed to the model's input layer
feature_columns = []

# set up the numerical columns for the cards' ranks
# with normalization function: (x-mean)/std
for i, header in enumerate(["C1", "C2", "C3", "C4", "C5"]):
    # set up numerical column
    num_col = tf.feature_column.numeric_column(header,
                                               normalizer_fn=lambda X:
                                               (X - X_means[i]) / X_stds[i])
    # append column to list
    feature_columns.append(num_col)

# set up the bucket columns for the cards' suits
for header in ["S1", "S2", "S3", "S4", "S5"]:

    # set up bucket column
    num_col = tf.feature_column.numeric_column(header)
    bucket_col = tf.feature_column.bucketized_column(num_col,
                                                     boundaries=list(
                                                         range(2, 5)))
    # append column to list
    feature_columns.append(bucket_col)
\`\`\`

#### Building the model
Now let's set up an initial model with Keras' seqential API, without thinking too much about parameters yet. We'll use 2 hidden layers, 200 units per layer, with exponential linear unit (elu) activation functions and He-initialized weights. After each hidden layer, we'll apply batch normalization to prevent vanishing or exploding gradients during training. We can also implement a early-stopping callback which will act against overfitting, as it just aborts training when the validation loss begins to increase again.


\`\`\`python
# bulding the sequential model
model = keras.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(200, activation='elu',kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(200, activation='elu',kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

# model checkpoint call back to save checkpints during training
checkpoint_cb = keras.callbacks.ModelCheckpoint("training/poker_hand_model.ckpt",
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1)

# early stopping callback to prevent overfitting on the  training data
earlystop_cb = keras.callbacks.EarlyStopping(patience=3,
                                             min_delta=0.01,
                                             restore_best_weights=True,
                                             verbose=1)


# Compile the model with a Nadam optimizer, initial learning rate 
# of 0.01 is relatively high.
# Since classes are given in one vector as integers from 0 to 9,
# we have to use sparse_categorical_crossentropy instead of
# categorical_crossentropy as the loss function.
model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.01),
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])

# fit the model with trianing and validation data, considering class weights 
# and early stopping (hence 100 epochs probably wont run)
history = model.fit(train_ds,
          steps_per_epoch=len(train) // batch_size,
          validation_data=valid_ds,
          validation_steps=len(valid) // batch_size,
          class_weight=class_weights,
          epochs=100,
          callbacks=[earlystop_cb,checkpoint_cb])
\`\`\`

    Train for 20000 steps, validate for 5000 steps
    Epoch 1/100
    19994/20000 [============================>.] - ETA: 0s - loss: 0.8818 - acc: 0.5876
    Epoch 00001: val_loss improved from inf to 0.80858, saving model to training/poker_hand_model.ckpt
    20000/20000 [==============================] - 130s 7ms/step - loss: 0.8817 - acc: 0.5877 - val_loss: 0.8086 - val_acc: 0.6579
    Epoch 2/100
    19998/20000 [============================>.] - ETA: 0s - loss: 0.6582 - acc: 0.7131
    Epoch 00002: val_loss did not improve from 0.80858
    20000/20000 [==============================] - 115s 6ms/step - loss: 0.6582 - acc: 0.7131 - val_loss: 1.1534 - val_acc: 0.6660
    Epoch 3/100
    19999/20000 [============================>.] - ETA: 0s - loss: 0.3995 - acc: 0.8401
    Epoch 00003: val_loss improved from 0.80858 to 0.77816, saving model to training/poker_hand_model.ckpt
    20000/20000 [==============================] - 121s 6ms/step - loss: 0.3995 - acc: 0.8401 - val_loss: 0.7782 - val_acc: 0.8856
    Epoch 4/100
    19994/20000 [============================>.] - ETA: 0s - loss: 0.3132 - acc: 0.8805
    Epoch 00004: val_loss improved from 0.77816 to 0.31105, saving model to training/poker_hand_model.ckpt
    20000/20000 [==============================] - 130s 7ms/step - loss: 0.3132 - acc: 0.8805 - val_loss: 0.3110 - val_acc: 0.9051
    Epoch 5/100
    19998/20000 [============================>.] - ETA: 0s - loss: 0.2695 - acc: 0.8996
    Epoch 00005: val_loss did not improve from 0.31105
    20000/20000 [==============================] - 118s 6ms/step - loss: 0.2695 - acc: 0.8996 - val_loss: 1.1440 - val_acc: 0.7433
    Epoch 6/100
    19989/20000 [============================>.] - ETA: 0s - loss: 0.2236 - acc: 0.9187
    Epoch 00006: val_loss did not improve from 0.31105
    20000/20000 [==============================] - 123s 6ms/step - loss: 0.2235 - acc: 0.9187 - val_loss: 8.0383 - val_acc: 0.8846
    Epoch 7/100
    19993/20000 [============================>.] - ETA: 0s - loss: 0.2006 - acc: 0.9282Restoring model weights from the end of the best epoch.
    
    Epoch 00007: val_loss did not improve from 0.31105
    20000/20000 [==============================] - 120s 6ms/step - loss: 0.2006 - acc: 0.9282 - val_loss: 5.8794 - val_acc: 0.9416
    Epoch 00007: early stopping
    

Since the training took a long time we better want to save (or load) the model in a serialized form.


\`\`\`python
# save the model (in TensorFlow's serialized SavedModel format, 
# you can also save it in HDF5 format by adding the file ending .h5)
model.save("saved_model/poker_hand_keras_model")

# load the model again
model = keras.models.load_model("saved_model/poker_hand_keras_model")
\`\`\`

We can evaluate the model on the test set.


\`\`\`python
# evaluate model on test set
loss, accuracy = model.evaluate(test_ds,steps=len(test)//batch_size)
print("Accuracy: ", accuracy, "\\nLoss: ", loss)
\`\`\`

    6250/6250 [==============================] - 16s 3ms/step - loss: 0.2396 - acc: 0.9068
    Accuracy:  0.906845 
    Loss:  0.23963412045776844
    

Wow, 95% accuracy, not bad for the first shot. Let's make predictions for the first 10 instances of our test set. We have a look at the targets first.


\`\`\`python
test.iloc[0:10,:].target.values
\`\`\`




    array([2, 1, 0, 1, 1, 1, 0, 0, 0, 1])




\`\`\`python
# get the first 10 instances from the test data and convert them to data set format
new_data = df_to_dataset(test.iloc[0:10,:],shuffle=False,batch_size=10)

# predict class probability
pred_proba = model.predict(new_data,steps=1) 

# predict classes
pred_calsses = np.argmax(pred_proba,axis=1)

pred_calsses
\`\`\`




    array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])



If we have a look at the probabilities predicted for the classes, we can see that the model is actually not very confident in it's prediction.


\`\`\`python
np.round(pred_proba[0],decimals=3)
\`\`\`




    array([0.   , 0.751, 0.248, 0.001, 0.   , 0.   , 0.   , 0.   , 0.   ,
           0.   ], dtype=float32)



It would be nice if we could be more certain for the minority classes as well. Accuracy is generally not a very god metric for problems with a high class imbalance. The macro-averaged F1 score may be a better metric to optimize for if we want to achieve a good classification for all classes. Furthermore, we've just blindly guessed how many layers and units per layer and so on we use in the model. We can probably do even better by optimizing the hyper-parameters. That's what we'll do in the next section.

#### Hyper-parameter optimization

For hyper-parameter optimization we will use the \`hyperas\` library, which is an easy-to-use Keras wrapper of the \`hyperopt\` library. To perform the optimization, we will first have to define functions which provide the training and validation data and build the model. I didn't manage to get the data function take global variables, \`hyperas\` would keep throwing errors; hence, I provided the entire data preprocessing again in this data function.


\`\`\`python
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform
from sklearn.metrics import f1_score



# function that provides the data for the optimization
def data():
    """Data providing function"""
    
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    import warnings
    warnings.filterwarnings('ignore')
    
    # import data from CSV file
    df = pd.read_csv("poker-hand-testing.data",
                     names = ["S1","C1","S2","C2","S3","C3",
                              "S4","C4","S5","C5","target"])
    
    # order the columns to have the ranks (C, numerical) 
    # and suits (S, categorical) together
    df = df[["C1","C2","C3","C4","C5",
             "S1","S2","S3","S4","S5",
             "target"]]
    
    # one-hot encode the suits features
    df = pd.get_dummies(df,columns=["S1","S2","S3","S4","S5"])

    
    # split off a test set and stratify for classes (target)
    train_full, test = train_test_split(df,
                                        stratify = df["target"],
                                        test_size=0.2, 
                                        random_state=123)
    
    # split into a training and a validation set and stratify for classes (target)
    train, valid = train_test_split(train_full,
                                    stratify = train_full["target"],
                                    test_size=0.2,
                                    random_state=123)
    
    # compute class weigths which will be used to account for class imbalance
    class_weights = compute_class_weight("balanced",
                                         np.unique(train["target"]),
                                         train["target"])
    
    # split features and labels
    X_train, y_train = train.drop("target",axis=1), train["target"].values
    X_valid, y_valid = valid.drop("target",axis=1), valid["target"].values
    X_test, y_test = test.drop("target",axis=1), test["target"].values
    
    # get rank and suit column names
    suit_cols = X_train.columns.drop(["C1","C2","C3","C4","C5"])
    rank_cols = ["C1","C2","C3","C4","C5"]
    
    # set up the preprocessor wit a scaler for the ranks
    preprocess = ColumnTransformer(transformers=[
        ("std",StandardScaler(),rank_cols),
        ("pass","passthrough",suit_cols)])
    
    # scale the rank values with a standard scaler
    X_train = preprocess.fit_transform(X_train)
    X_valid = preprocess.transform(X_valid)
    X_test = preprocess.transform(X_test)
        
    
    
    return X_train, y_train, X_vaild, y_vaild, X_test, y_test, class_weights
    
# function that builds the model for the optimization
def build_model(X_train, y_train, X_vaild, y_vaild, X_test, y_test, class_weights):
    """Model providing function"""    
    
    # parameters to optimize
    num_layers = {{choice([2,3,4])}}
    num_units = int({{uniform(50,400)}})
    learning_rate = {{loguniform(-8,-4)}}
    batch_size=32
    
    print(f"New parameters:\\nNumber of layers: {num_layers}\\nNumber of units: {num_units}\\nLearning rate: {learning_rate}")
    
    # create model and add input layer with feature columns
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train.shape[1:],batch_size=batch_size))
    
    # add hidden layer
    for i in range(num_layers):
        model.add(keras.layers.Dense(units=num_units,
                                     activation='elu',
                                     kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
    
    # add ourput layer
    model.add(keras.layers.Dense(units=10, 
                                 activation='softmax',))
    
    # compile model with optimizer
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    
    # fitthe model with class weights applied
    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              validation_data=(X_valid,y_valid),
              class_weight=class_weights,
              epochs=1,
              verbose=0)
    
    # make class predictions with validation data
    pred_proba = model.predict(X_valid) 
    pred_calsses = np.argmax(pred_proba,axis=1)
    
    # calculate the macro-averaged F1 score based on the predictions
    f1 = f1_score(y_valid,pred_calsses,average="macro")
    
    # print results of the optimization round
    print(f"F1 score: {f1:.3f}\\nfor\\nNumber of layers: {num_layers}\\nNumber of units: {num_units}\\nLearning rate: {learning_rate}")
    
    
    return {'loss': -f1, 'status': STATUS_OK, 'model': model}

\`\`\`

    Using TensorFlow backend.
    

Now that the functions are set up, let's perform 30 optimization steps.


\`\`\`python
# perform a hyperas optimization
best_run, best_model = optim.minimize(model=build_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=30,
                                      trials=Trials(),
                                      notebook_name='Deep_learning_poker_hand')
\`\`\`

Let's have a look at the best parameters we found. 


\`\`\`python
best_run
\`\`\`


\`\`\`python
best_model.summary()
\`\`\`

Looks like we should train a model with 3 hidden layers, 265 units per layer, and a learning rate of 0.12. Let's do that and see how good we can get.


\`\`\`python
def build_opt_model():
    # bulding the sequential model
    opt_model = keras.Sequential([
        keras.layers.DenseFeatures(feature_columns),
        keras.layers.Dense(265, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(265, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(265, activation='elu', kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # model checkpoint call back to save checkpints during training
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "training/poker_hand_model.ckpt",
        save_best_only=True,
        save_weights_only=True,
        verbose=1)
    
    # early stopping callback to prevent overfitting on the  training data
    earlystop_cb = keras.callbacks.EarlyStopping(patience=5,
                                                 min_delta=0.01,
                                                 restore_best_weights=True,
                                                    verbose=1)
    
    # Compile the model with a Nadam optimizer    
    opt_model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.012),
                      loss="sparse_categorical_crossentropy",
                      metrics=["acc"])
    
    return opt_model, checkpoint_cb, earlystop_cb
    
opt_model, checkpoint_cb, earlystop_cb = build_opt_model()

# fit the model with trianing and validation data, considering class weights
# and early stopping (hence 100 epochs probably wont run)
history = opt_model.fit(train_ds,
                        steps_per_epoch=len(train) // batch_size,
                        validation_data=valid_ds,
                        validation_steps=len(valid) // batch_size,
                        class_weight=class_weights,
                        epochs=100,
                        callbacks=[earlystop_cb, checkpoint_cb])
\`\`\`

    Train for 20000 steps, validate for 5000 steps
    Epoch 1/100
    19996/20000 [============================>.] - ETA: 0s - loss: 0.8743 - acc: 0.5968
    Epoch 00001: val_loss improved from inf to 0.68111, saving model to training/poker_hand_model.ckpt
    20000/20000 [==============================] - 189s 9ms/step - loss: 0.8742 - acc: 0.5969 - val_loss: 0.6811 - val_acc: 0.7055
    Epoch 2/100
    19998/20000 [============================>.] - ETA: 0s - loss: 0.4881 - acc: 0.8041
    Epoch 00002: val_loss did not improve from 0.68111
    20000/20000 [==============================] - 178s 9ms/step - loss: 0.4881 - acc: 0.8041 - val_loss: 1.3419 - val_acc: 0.6085
    Epoch 3/100
    19996/20000 [============================>.] - ETA: 0s - loss: 0.3234 - acc: 0.8769
    Epoch 00003: val_loss did not improve from 0.68111
    20000/20000 [==============================] - 191s 10ms/step - loss: 0.3234 - acc: 0.8769 - val_loss: 98.4170 - val_acc: 0.8010
    Epoch 4/100
    19993/20000 [============================>.] - ETA: 0s - loss: 0.2656 - acc: 0.9019
    Epoch 00004: val_loss did not improve from 0.68111
    20000/20000 [==============================] - 182s 9ms/step - loss: 0.2656 - acc: 0.9019 - val_loss: 1.0344 - val_acc: 0.6891
    Epoch 5/100
    19994/20000 [============================>.] - ETA: 0s - loss: 0.2323 - acc: 0.9150
    Epoch 00005: val_loss did not improve from 0.68111
    20000/20000 [==============================] - 182s 9ms/step - loss: 0.2323 - acc: 0.9150 - val_loss: 0.8909 - val_acc: 0.8548
    Epoch 6/100
    19996/20000 [============================>.] - ETA: 0s - loss: 0.2076 - acc: 0.9251Restoring model weights from the end of the best epoch.
    
    Epoch 00006: val_loss did not improve from 0.68111
    20000/20000 [==============================] - 186s 9ms/step - loss: 0.2076 - acc: 0.9251 - val_loss: 1609.0826 - val_acc: 0.8910
    Epoch 00006: early stopping
    

Let's save the model again see how we perform on the F1 score now.


\`\`\`python
# save the model
opt_model.save("saved_model/poker_hand_keras_opt_model")

# load the model again
opt_model = keras.models.load_model("saved_model/poker_hand_keras_opt_model")
\`\`\`

    WARNING:tensorflow:From /home/pascal/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1781: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: saved_model/poker_hand_keras_opt_model/assets
    


\`\`\`python
from sklearn.metrics import f1_score

# make predictions for the test set (steps = number of instances / batch size)
pred_proba = opt_model.predict(test_ds,steps=6250) 
pred_calsses = np.argmax(pred_proba,axis=1)
    
# calculate the macro-averaged F1 score based on the predictions
f1_macro = f1_score(test["target"].values,pred_calsses,average="macro")
f1_micro = f1_score(test["target"].values,pred_calsses,average="micro")

print(f"F1-macro: {f1_macro:.2f}\\nF1-micro: {f1_micro:.2f}")
\`\`\`

    F1-macro: 0.19
    F1-micro: 0.71
    

We can see that the micro-averaged F1 score, which considers all instances individually, is not too bad. But the macro-averaged F1 score, which is an unweighted average of every classes' F1 score, is much worse. This clearly shows that we still have a big problem with classifying the strongly underrepresented classes correctly. This becomes clear when we look at the confusion matrix below. 


\`\`\`python
from sklearn.metrics import confusion_matrix
confusion_matrix(test["target"].values,pred_calsses)
\`\`\`




    array([[95597,  4644,     1,     0,     0,     0,     0,     0,     0,
                0],
           [40088, 44265,    53,    94,     0,     0,     0,     0,     0,
                0],
           [  538,  8284,    63,   639,     0,     0,     0,     0,     0,
                0],
           [   10,  2696,    60,  1458,     0,     0,     0,     0,     0,
                0],
           [  427,   350,     0,     0,     0,     0,     0,     0,     0,
                0],
           [  393,     6,     0,     0,     0,     0,     0,     0,     0,
                0],
           [    0,    74,     2,   209,     0,     0,     0,     0,     0,
                0],
           [    0,     0,     0,    42,     0,     0,     3,     1,     0,
                0],
           [    2,     0,     0,     0,     0,     0,     0,     0,     0,
                0],
           [    1,     0,     0,     0,     0,     0,     0,     0,     0,
                0]])



#### Resampling the data set to counter class imbalance
As we have seen above, the neural network has trouble classifying the underrepresented classes correctly. One approach we can try is to bootstrap (sample with replacement) a new dataset from the original data set in which the minority classes are oversampled so that all classes are balanced. We don't actually provide any new data but since training a neural network is a stochastic process, we can try to reduce its bias towards the majority class with this method.

Let's bootstrap 100000 samples for each of the classes.


\`\`\`python
# create data frame with 1,000,000 bootstrapped instances, 100,000 from each class
df_resampled = pd.concat([
    df.loc[df.target == label, :].sample(100000, replace=True)
    for label in df.target.unique()
])
\`\`\`

And split it into train, validation, and test sets again. The suffix _rs stands for resampled.


\`\`\`python
# split off a test set and stratify for classes (target)
train_full_rs, test_rs = train_test_split(df_resampled,
                                          stratify=df_resampled["target"],
                                          test_size=0.2,
                                          random_state=123)

# split into a training and a validation set and stratify for classes (target)
train_rs, valid_rs = train_test_split(train_full_rs,
                                      stratify=train_full_rs["target"],
                                      test_size=0.2,
                                      random_state=123)
\`\`\`

Recompute column means and standard deviations (due to the oversampling of the minority classes, these quantities are quite a bit different now).


\`\`\`python
# calculate means and standard deviations for the numerical features
X_means = tf.math.reduce_mean(train_rs.iloc[:, :5].values.astype(np.float32),
                              axis=0)
X_stds = tf.math.reduce_std(train_rs.iloc[:, :5].values.astype(np.float32),
                            axis=0)
print(X_means, X_stds)
\`\`\`

    tf.Tensor([7.245467  7.151083  7.5978374 7.1570797 7.494875 ], shape=(5,), dtype=float32) tf.Tensor([3.775464  3.740848  3.7774026 3.7822576 3.6998467], shape=(5,), dtype=float32)
    

Convert the data to \`TensorFlow\` data set format.


\`\`\`python
batch_size = 32
# create tensorflow data sets from data frames
train_rs_ds = df_to_dataset(train_rs, batch_size=batch_size)
valid_rs_ds = df_to_dataset(valid_rs, shuffle=False, batch_size=batch_size)
test_rs_ds = df_to_dataset(test_rs, shuffle=False, batch_size=batch_size)
\`\`\`

Train a new model with the resampled data.


\`\`\`python
opt_model_rs, checkpoint_cb, earlystop_cb = build_opt_model()

# fit the model with trianing and validation data, considering class weights
# and early stopping (hence 100 epochs probably wont run)
history_rs = opt_model_rs.fit(train_rs_ds,
                              steps_per_epoch=len(train_rs) // batch_size,
                              validation_data=valid_rs_ds,
                              validation_steps=len(valid_rs) // batch_size,
                              epochs=100,
                              callbacks=[earlystop_cb, checkpoint_cb])
\`\`\`

    Train for 20000 steps, validate for 5000 steps
    Epoch 1/100
    19993/20000 [============================>.] - ETA: 0s - loss: 0.7889 - acc: 0.6875
    Epoch 00001: val_loss improved from inf to 0.38209, saving model to training/poker_hand_model.ckpt
    20000/20000 [==============================] - 213s 11ms/step - loss: 0.7888 - acc: 0.6876 - val_loss: 0.3821 - val_acc: 0.8474
    Epoch 2/100
    19998/20000 [============================>.] - ETA: 0s - loss: 0.3663 - acc: 0.8611
    Epoch 00002: val_loss did not improve from 0.38209
    20000/20000 [==============================] - 149s 7ms/step - loss: 0.3663 - acc: 0.8611 - val_loss: 0.9238 - val_acc: 0.9216
    Epoch 3/100
    19993/20000 [============================>.] - ETA: 0s - loss: 0.2054 - acc: 0.9277
    Epoch 00003: val_loss did not improve from 0.38209
    20000/20000 [==============================] - 142s 7ms/step - loss: 0.2054 - acc: 0.9278 - val_loss: 184.8528 - val_acc: 0.9716
    Epoch 4/100
    19992/20000 [============================>.] - ETA: 0s - loss: 0.1384 - acc: 0.9538
    Epoch 00004: val_loss did not improve from 0.38209
    20000/20000 [==============================] - 153s 8ms/step - loss: 0.1384 - acc: 0.9538 - val_loss: 8.2383 - val_acc: 0.9768
    Epoch 5/100
    19999/20000 [============================>.] - ETA: 0s - loss: 0.1036 - acc: 0.9666
    Epoch 00005: val_loss did not improve from 0.38209
    20000/20000 [==============================] - 182s 9ms/step - loss: 0.1036 - acc: 0.9666 - val_loss: 15.8309 - val_acc: 0.9879
    Epoch 6/100
    19999/20000 [============================>.] - ETA: 0s - loss: 0.0878 - acc: 0.9723Restoring model weights from the end of the best epoch.
    
    Epoch 00006: val_loss did not improve from 0.38209
    20000/20000 [==============================] - 175s 9ms/step - loss: 0.0877 - acc: 0.9723 - val_loss: 433.6815 - val_acc: 0.9931
    Epoch 00006: early stopping
    


\`\`\`python
# save the model
opt_model_rs.save("saved_model/poker_hand_keras_opt_model_rs")

# load the model again
opt_model_rs = keras.models.load_model("saved_model/poker_hand_keras_opt_model_rs")
\`\`\`

    INFO:tensorflow:Assets written to: saved_model/poker_hand_keras_opt_model_rs/assets
    

Okay, now let's make some predictions again and look at the F1 scores.


\`\`\`python
# make predictions for the test set (steps = number of instances / batch size)
pred_proba = opt_model_rs.predict(test_rs_ds,steps=6250) 
pred_calsses = np.argmax(pred_proba,axis=1)
    
# calculate the macro-averaged F1 score based on the predictions
f1_macro = f1_score(test_rs["target"].values,pred_calsses,average="macro")
f1_micro = f1_score(test_rs["target"].values,pred_calsses,average="micro")

print(f"F1-macro: {f1_macro:.2f}\\nF1-micro: {f1_micro:.2f}")
confusion_matrix(test_rs["target"].values,pred_calsses)
\`\`\`

    F1-macro: 0.84
    F1-micro: 0.85
    




    array([[13296,  4792,   971,     5,   911,    25,     0,     0,     0,
                0],
           [ 2294,  9522,  4722,  1308,  2064,     8,    82,     0,     0,
                0],
           [   14,  2171, 11278,  2226,  2128,     0,  2141,    42,     0,
                0],
           [    0,    34,   303, 15868,    28,     0,  2648,  1119,     0,
                0],
           [   10,    72,    29,     0, 19889,     0,     0,     0,     0,
                0],
           [    0,     0,     0,     0,     0, 19943,     0,     0,    57,
                0],
           [    0,     0,     0,    36,     0,     0, 19784,   180,     0,
                0],
           [    0,     0,     0,     0,     0,     0,     0, 20000,     0,
                0],
           [    0,     0,     0,     0,     0,     0,     0,     0, 20000,
                0],
           [    0,     0,     0,     0,     0,     0,     0,     0,     0,
            20000]])



Now the micro and macro averaged F1 score are almost identical. The resampling has certainly helped with the class imbalance and slightly improved the overall performance. Looking at the confusion matrix, we can see that the very strongly oversampled minority classes are unsurprisingly all classified correctly because they're always the same reoccurring examples, whereas the original majority class still shows several misclassifications, most likely because not all possible examples of these classes were sampled from the original data. 

### ResNet CNN for classifiying cats and dogs <a id="catdog"></a>
This Asirra (Animal Species Image Recognition for Restricting Access) data set is a  HIP (Human Interactive Proof) that works by asking users to identify photographs of cats and dogs. It contains pictures of cats and dogs which can be used to train an image classifier. This time it does not originate from the UCI machine learning repository, but from a [Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) which was hosted in 2013. The Jupyter Notebook can be found [here](https://github.com/Pascal-Bliem/exploring-the-UCI-ML-repository/tree/master/ResNet-CNNs-on-cats-and-dogs).

Since my crappy old laptop only as a weak CPU but I want to use GPUs, I'm actually running this project on [Google colab](https://colab.research.google.com/).

#### Frame the problem
We have a balanced data set of 25000 images of cats and dogs to train a classifier which can tell them apart. While the number of images is not super small, it is still far from enough to easily train any kind of neural network. We will use convolutional neural networks (CNN) which recognize image features much more data-efficiently than fully connected networks. Do artificially enlarge the data set and counter overfitting, we will use data augmentation techniques to modify the images before we pass them into the CNN. Instead of freely experimenting with the CNN architecture, we will choose an architecture which has proven to work very well in past benchmark tests: the ResNet. Furthermore, we will try to train this network from scratch as well as using a network which was trained on the [ImageNet](http://www.image-net.org/) data set to see how much performance we can gain by using pretrained weights. Let's get started!


\`\`\`python
%pip install tensorflow-addons
%pip install tensorflow-gpu
# import the libraries we'll need
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import os
\`\`\`

#### Preparing the data
First we have to set up a pipeline to get the images and then prepare the image augmentation. Let's load the file paths and labels into a data frame first.


\`\`\`python
data_path = "./data/train/"
filenames = os.listdir(data_path)
labels = []
for filename in filenames:
    label = filename.split('.')[0]
    if label == 'dog':
        labels.append("dog")
    else:
        labels.append("cat")

df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})
\`\`\`


\`\`\`python
# let's look at a random image
plt.imshow(keras.preprocessing.image.load_img(data_path+df.iloc[100].filename))
\`\`\`




    <matplotlib.image.AxesImage at 0x7f1ff31426d8>




<img style="width: 60%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/catdog_output_9_1.png">


Yep, that looks like a cat. Let's first split off training, validation, and test.


\`\`\`python
train_df_full, test_df = train_test_split(df, test_size=0.20, random_state=42)
train_df, validate_df = train_test_split(train_df_full, test_size=0.25, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
\`\`\`

Let's define some constants which we'll use later.


\`\`\`python
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
CLASS_NAMES = np.array(["dog","cat"])
BATCH_SIZE=64
AUTOTUNE = tf.data.experimental.AUTOTUNE

# tell Tensorflow to use XLA (accelerated linear algebra)
tf.config.optimizer.set_jit(True)
\`\`\`

Now we can prepare a generator to stream from for training the model. Keras' image preprocessing utilities provide a function for this. For the training data, we'll implement the augmentation as well.


\`\`\`python
# The image generator for the training data will apply augmentation operations
# such as rotation, shear, zoom, shifting and horizontal flipping.
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# we will then let the generator stream the data from the filenames 
# stored in the data frames
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    data_path, 
    x_col="filename",
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)
\`\`\`

    Found 15000 validated image filenames belonging to 2 classes.
    

Let's have a look at the augmentations which the generator applies.


\`\`\`python
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    data_path, 
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
\`\`\`

    Found 1 validated image filenames belonging to 1 classes.
    


\`\`\`python
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
\`\`\`


![](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/catdog_output_18_0.png)


While using the Keras preprocessing utilities is very convenient, it is unfortunately not very fast and well-integrated with the TensorFlow ecosystem. Luckily, we can also apply these augmentation by using tensorflow functions and feeding the data into the model in the TensorFlow data set format.


\`\`\`python
# load the list of filenames as datasets
train_ds = tf.data.Dataset.from_tensor_slices(data_path + train_df.filename.values)
validate_ds = tf.data.Dataset.from_tensor_slices(data_path + validate_df.filename.values)
test_ds = tf.data.Dataset.from_tensor_slices(data_path + test_df.filename.values)


for f in train_ds.take(5):
  print(f.numpy())
\`\`\`

    b'./data/train/cat.8410.jpg'
    b'./data/train/dog.12008.jpg'
    b'./data/train/dog.6125.jpg'
    b'./data/train/cat.8437.jpg'
    b'./data/train/dog.5051.jpg'
    

Now we can implement the augmentation functions.


\`\`\`python
@tf.function(experimental_relax_shapes=True)
def augment(img):
    """Apply random horizontal flipping, rotation, shearing, 
    shifting, zooming, and change of brightness, contrats,
    and saturation. The scaling factors are all random,
    look at the individual lines to see the boundaries.
    
    Args:
        img: Image
    Returns:
        Augmented image
    """
        
    # horizontal flipping    
    img = tf.image.random_flip_left_right(img)
    
    # rotation
    img = tfa.image.rotate(img, tf.random.uniform((1,),-0.2,0.2)[0], interpolation='BILINEAR')
    
    # shearing
    shear_lambda = tf.random.uniform((1,),-0.1,0.1)[0]
    forward_transform = [[1.0,shear_lambda,0],[0,1.0,0],[0,0,1.0]]
    t = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
    img = tfa.image.transform(img, t, interpolation="BILINEAR")
    
    # shifting
    trans = tf.random.uniform((1,),-0.1,0.1)[0]
    img = tfa.image.translate(img, translations=[trans,trans])
    
    # zoom by cropping and resizing
    offset = tf.random.uniform((1,),0.0,0.1)[0]
    shift = tf.random.uniform((1,),0.9,1.0)[0]
    img_crp = tf.image.crop_and_resize(tf.reshape(img,[1,img.shape[0],img.shape[1],3]), 
                                       boxes=[[offset,offset,shift,shift]],
                                       box_indices=[0],
                                       crop_size=[img.shape[0], img.shape[1]])
    img = tf.reshape(img_crp,[img.shape[0],img.shape[1],3])
    
    # change brightness, contrast, and saturation
    img = tf.image.adjust_brightness(img, tf.random.uniform((1,),-0.2,0.2)[0])
    img = tf.image.adjust_contrast(img, contrast_factor=1+tf.random.uniform((1,),-0.1,0.1)[0])
    img = tf.image.adjust_saturation(img,1+tf.random.uniform((1,),-0.2,0.2)[0])
    
    return img


\`\`\`

Now we'll have to write some functions to get the labels and convert the images to tensors.


\`\`\`python
@tf.function(experimental_relax_shapes=True)
def get_label(file_path):
  # convert the path to a list of path components
    parts = tf.strings.split(file_path, "/")
    parts = tf.strings.split(parts[-1], ".")
    # note that we output [[1],[0]] for dog and [[0],[1]] for cat
    # because we will fit with binary cross entropy loss.
    # we could also output [[1]] or [[0]] respectively,
    # if we use sparse categorical cross entropy
    if parts[0] == "dog":
        return np.array([1,0]).reshape(-1,1)
    else:
        return np.array([0,1]).reshape(-1,1)

@tf.function    
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use \`convert_image_dtype\` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])

@tf.function
def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

@tf.function
def process_path_aug(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = augment(img)
    return img, label

train_aug_ds = train_ds.map(process_path_aug, num_parallel_calls=AUTOTUNE)
train_noaug_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
validate_ds = validate_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
\`\`\`

Okay, let's have a look if tigs are in the right shape.


\`\`\`python
for img, label in train_aug_ds.take(2):
  print("Image shape: ", img.numpy().shape)
  print("Label: ", label.numpy())
\`\`\`

    Image shape:  (224, 224, 3)
    Label:  [[0]
     [1]]
    Image shape:  (224, 224, 3)
    Label:  [[1]
     [0]]
    

And let's see how the augmented vs. unaugmented image looks like.


\`\`\`python
for img, label in train_aug_ds.take(1):
        img1 = img
for img, label in train_noaug_ds.take(1):
        img2 = img
        
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img1)
plt.subplot(1,2,2)
plt.imshow(img2)
plt.show()
\`\`\`

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

<img style="width: 70%;" src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/exploring-the-UCI-ML-repository/catdog_output_28_1.png">

Now the last step is to prepare the data sets for training by shuffling, batching, and prefetching.


\`\`\`python
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    #  If the data set is too large to fit in memory then use 
    # \`.cache(filename)\` to cache preprocessing work for datasets 
    #  that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
    else:
        ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  
    # Repeat forever
    ds = ds.repeat()
  
    ds = ds.batch(BATCH_SIZE)
  
    # \`prefetch\` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
  
    return ds

train_aug_ds = prepare_for_training(train_aug_ds)
train_noaug_ds = prepare_for_training(train_noaug_ds)
validate_ds = prepare_for_training(validate_ds)
test_ds = prepare_for_training(test_ds)
\`\`\`

#### Building the models
We can finally focus on the model. As already mentioned in the title of this notebook, we want to use a ResNet or residual network, developed by [He et al.](https://github.com/KaimingHe/deep-residual-networks). This type of network won the ILSVRC 2015 challenge with a top-5 error rate under 3.6%. The special design of this network includes residual units with skip connections. These connections basically skip a layer of the network and feed the signal directly into a layer that is higher up in the stack. Why is that useful at all? When a network is initialized with weights close to zero, it will also output values close to zero. Layers that have not started learning can block the flow of backpropagation in the network which makes it difficult to train very deep networks. With skip connections, however, the network can just output its input and easily propagate a signal.

The more parameters we want to learn, the more data we need. Since we don't have an awful lot of data, we'll use a smaller ResNet architecture, the ResNet-34 for the model we're building from scratch. Later we'll compare it to a pretrained ResNet-50. Let's build a class for the residual units first.


\`\`\`python
from functools import partial

# This will be the default convolutional layer with a 3x3 filter moving with a stride of 1 and
# applying SAME padding (padding the input with zeroes so that the output has the same shape)
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,padding="SAME", use_bias=False)

# this will be the class for a residual unit layer
class ResidualUnit(keras.layers.Layer):

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        
        # The main layers: 2 convolutional layers (the first one may have a larger stride),
        # with batch normalization after each layer
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        
        # if the stride is 2, the output dimensions will be halved so 
        # we need a 1x1 convolutional layer with a stride of 2 to 
        # adjust the output dimensions of the skip connection
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]
    
    def call(self, inputs):
        # propagate the input through the main and skip layers (if present)
        # and return the sum of the two outputs through the activation fucntion
        main = inputs
        for layer in self.main_layers:
            main = layer(main)
        
        skip = inputs
        for layer in self.skip_layers:
            skip = layer(skip)
        
        return self.activation(main + skip)
\`\`\`

Now we can use this residual unit to build a residual network with it.


\`\`\`python
# get the input shape
inp, _ = next(iter(train_aug_ds.take(1)))
input_shape = inp.numpy().shape[1:]

# set up the model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))

# initial convolutional layer
model.add(DefaultConv2D(64, kernel_size=7, strides=2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

# Every few steps we go deeper down the network, we will double the
# number of filter maps and reduce the dimensions of the output by
# applying a stride of 2. 
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
    
# apply global average pooling and feed the output directly into 
# the output layer which has a sigmoid activation
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2, activation="sigmoid"))

# compile the model with Nadam optimizer
optimizer = keras.optimizers.Nadam()
model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])

\`\`\`


\`\`\`python
history = model.fit(train_aug_ds, 
                        steps_per_epoch=15000//BATCH_SIZE,
                        validation_data=validate_ds,
                        validation_steps=5000//BATCH_SIZE,
                        epochs=1)
\`\`\`

    Train for 234 steps, validate for 78 steps
    234/234 [==============================] - 295s 1s/step - loss: 0.6947 - accuracy: 0.6079 - val_loss: 0.7266 - val_accuracy: 0.5571
    


\`\`\`python
model.save("ResNet34_save")
\`\`\`

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1781: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: ResNet34_save/assets
    

Okay, only one period didn't give us amazing accuracy. Let's train the model until we don't gain any additional validation loss improvement by implementing an early stopping callbak.


\`\`\`python
# early stopping callback to prevent overfitting on the  training data
earlystop_cb = keras.callbacks.EarlyStopping(patience=2,
                                             min_delta=0.01,
                                             restore_best_weights=True,
                                             verbose=1)
history2 = model.fit(train_aug_ds, 
                     steps_per_epoch=15000//BATCH_SIZE,
                     validation_data=validate_ds,
                     validation_steps=5000//BATCH_SIZE,
                    epochs=100,callbacks=[earlystop_cb])
\`\`\`

    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 262s 1s/step - loss: 0.6049 - accuracy: 0.6701 - val_loss: 0.8103 - val_accuracy: 0.5921
    Epoch 2/100
    234/234 [==============================] - 248s 1s/step - loss: 0.5449 - accuracy: 0.7241 - val_loss: 0.6400 - val_accuracy: 0.6834
    Epoch 3/100
    234/234 [==============================] - 249s 1s/step - loss: 0.4874 - accuracy: 0.7682 - val_loss: 0.8053 - val_accuracy: 0.6216
    Epoch 4/100
    233/234 [============================>.] - ETA: 0s - loss: 0.4247 - accuracy: 0.8076Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 251s 1s/step - loss: 0.4243 - accuracy: 0.8079 - val_loss: 0.7702 - val_accuracy: 0.7420
    Epoch 00004: early stopping
    


\`\`\`python
model.save("ResNet34_save")
\`\`\`

That's pretty great already, we got a validation accuracy of about 75%. Now let's see if we can even improve that by using a ResNet-50 pretrained on the ImageNet data set. Of course we'll have to remove the top layer of this pretrained model and add the one suitable for our classification task. We'll first freeze the weights of the lower layers, train the upper layers, and then gradually try to unfreeze some of the lower layers.


\`\`\`python
# get the pretrained ResNet50
base_model = keras.applications.resnet.ResNet50(weights="imagenet",include_top=False)
\`\`\`


\`\`\`python
base_model = keras.applications.resnet.ResNet50(weights="imagenet",include_top=False)
# now we'll add global average pooling and the output layer on top of the base model
avgpool = keras.layers.GlobalAvgPool2D()(base_model.output)
flatten = keras.layers.Flatten()(avgpool)
output = keras.layers.Dense(2, activation="sigmoid")(flatten)
model2 = keras.models.Model(inputs=base_model.input, outputs=output)

# freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# compile the model with Nadam optimizer and dynamic loss scale
model2.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
\`\`\`

Okay, let's train it.


\`\`\`python
# early stopping callback to prevent overfitting on the  training data
earlystop_cb = keras.callbacks.EarlyStopping(patience=2,
                                             min_delta=0.01,
                                             restore_best_weights=True,
                                             verbose=1)

history3 = model2.fit(train_aug_ds, 
                    steps_per_epoch=15000//BATCH_SIZE,
                    validation_data=validate_ds,
                    validation_steps=5000//BATCH_SIZE,
                    epochs=100,callbacks=[earlystop_cb])
\`\`\`

    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 276s 1s/step - loss: 0.1193 - accuracy: 0.9547 - val_loss: 1.4036 - val_accuracy: 0.5098
    Epoch 2/100
    234/234 [==============================] - 246s 1s/step - loss: 0.0752 - accuracy: 0.9704 - val_loss: 1.6608 - val_accuracy: 0.5090
    Epoch 3/100
    233/234 [============================>.] - ETA: 0s - loss: 0.0705 - accuracy: 0.9735Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 245s 1s/step - loss: 0.0707 - accuracy: 0.9735 - val_loss: 1.4818 - val_accuracy: 0.5094
    Epoch 00003: early stopping
    

Mhh, the model severly overfitted on the training data. Let's try to unfreeze the layers and introduce a dropout layer.


\`\`\`python
base_model = keras.applications.resnet.ResNet50(weights="imagenet",include_top=False)
# now we'll add global average pooling and the output layer on top of the base model
dropout1 = keras.layers.Dropout(0.5)(base_model.output)
avgpool = keras.layers.GlobalAvgPool2D()(dropout1)
flatten = keras.layers.Flatten()(avgpool)
dropout2 = keras.layers.Dropout(0.5)(flatten)
output = keras.layers.Dense(2, activation="sigmoid")(dropout2)
model3 = keras.models.Model(inputs=base_model.input, outputs=output)

# freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-31:]:
    layer.trainable = True

# compile the model with Nadam optimizer and dynamic loss scale
model3.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])

earlystop_cb = keras.callbacks.EarlyStopping(patience=2,
                                             min_delta=0.01,
                                             restore_best_weights=True,
                                             verbose=1)

history4 = model3.fit(train_aug_ds, 
                    steps_per_epoch=15000//BATCH_SIZE,
                    validation_data=validate_ds,
                    validation_steps=5000//BATCH_SIZE,
                    epochs=100,callbacks=[earlystop_cb])
\`\`\`

    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 281s 1s/step - loss: 0.0807 - accuracy: 0.9694 - val_loss: 0.6941 - val_accuracy: 0.5021
    Epoch 2/100
    234/234 [==============================] - 250s 1s/step - loss: 0.0461 - accuracy: 0.9820 - val_loss: 0.6952 - val_accuracy: 0.4909
    Epoch 3/100
    233/234 [============================>.] - ETA: 0s - loss: 0.0361 - accuracy: 0.9863Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 250s 1s/step - loss: 0.0361 - accuracy: 0.9863 - val_loss: 0.6972 - val_accuracy: 0.4895
    Epoch 00003: early stopping
    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 281s 1s/step - loss: 0.0807 - accuracy: 0.9694 - val_loss: 0.6941 - val_accuracy: 0.5021
    Epoch 2/100
    234/234 [==============================] - 250s 1s/step - loss: 0.0461 - accuracy: 0.9820 - val_loss: 0.6952 - val_accuracy: 0.4909
    Epoch 3/100
    233/234 [============================>.] - ETA: 0s - loss: 0.0361 - accuracy: 0.9863Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 250s 1s/step - loss: 0.0361 - accuracy: 0.9863 - val_loss: 0.6972 - val_accuracy: 0.4895
    Epoch 00003: early stopping
    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 281s 1s/step - loss: 0.0807 - accuracy: 0.9694 - val_loss: 0.6941 - val_accuracy: 0.5021
    Epoch 2/100
    234/234 [==============================] - 250s 1s/step - loss: 0.0461 - accuracy: 0.9820 - val_loss: 0.6952 - val_accuracy: 0.4909
    Epoch 3/100
    233/234 [============================>.] - ETA: 0s - loss: 0.0361 - accuracy: 0.9863Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 250s 1s/step - loss: 0.0361 - accuracy: 0.9863 - val_loss: 0.6972 - val_accuracy: 0.4895
    Epoch 00003: early stopping
    Train for 234 steps, validate for 78 steps
    Epoch 1/100
    234/234 [==============================] - 281s 1s/step - loss: 0.0807 - accuracy: 0.9694 - val_loss: 0.6941 - val_accuracy: 0.5021
    Epoch 2/100
    234/234 [==============================] - 250s 1s/step - loss: 0.0461 - accuracy: 0.9820 - val_loss: 0.6952 - val_accuracy: 0.4909
    Epoch 3/100
    233/234 [============================>.] - ETA: 0s - loss: 0.0361 - accuracy: 0.9863Restoring model weights from the end of the best epoch.
    234/234 [==============================] - 250s 1s/step - loss: 0.0361 - accuracy: 0.9863 - val_loss: 0.6972 - val_accuracy: 0.4895
    Epoch 00003: early stopping
    

Okay, look like reusing a pretrained model doesn't work well here. The overfitting is too severe. But hey, the model we trained from scrathc was already doing pretty well, so kind of a success :)

### The End
Okay, that is enough now. I hope you enjoyed going through five less common data sets and examples for their applications. Maybe you even learned something new about machine learning. Thanks for making it all the way through till here!
`
);
