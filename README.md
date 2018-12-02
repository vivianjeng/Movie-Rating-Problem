# Movie Rating Problem

## Requirement
 - python3
 - keras >= 2.1.5
 - sklearn >= 0.0


## Files
 - `src/predict.py`, builds predicting/embedding model
 - `src/analysis.py`, uses TSNE and KMeans to classify the movies
 - `Movie-Rating-Problem-鄭雅文.pdf`, the report
 - `figures/`, the figures of the report
 - `output data/`, the csv files for analysis


## Usage

`
python3 src/predict.py [rating file]
`
  
It will generate a file named 'testing.h5' (keras model) and a file named 'predict.csv' (predicted rating).

`
python3 src/analysis.py
`
  
It will do classification.

## Author

Jeng, Ya-wen (鄭雅文)
r06922097@ntu.edu.tw

## References
 - [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/), movie dataset
 - [Plotly](https://plot.ly/create/), graph maker
