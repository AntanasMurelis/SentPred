# SentPred: COVID-19 Sentiment Analysis Across US

## Project Overview

This project is an exploration of public sentiment in the United States during the COVID-19 pandemic. We analyzed a dataset of tweets from across the country, using BERT representations to understand the sentiment expressed in each tweet. The result is a comprehensive view of how sentiment has evolved throughout the pandemic across different states.

## Data

The primary dataset used for this project consists of tweets collected over the duration of the COVID-19 pandemic. Each tweet is associated with a specific timestamp and a US state. The collection of these tweets offers a rich source of data that reflects public sentiment during this unprecedented period.

## BERT Representations

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art NLP model that Google developed and open-sourced. It's particularly well-suited for understanding the context of a word in a sentence, which is essential for accurate sentiment analysis.

We used BERT to convert each tweet into a vector representation. These vectors capture the semantic meaning of the tweet, allowing us to analyze them in a way that traditional text analysis techniques can't achieve.

![](https://github.com)

## Sentiment Analysis

The vector representations from BERT were then fed into a sentiment analysis model. This model classifies each tweet into one of three sentiment classes: -1 (negative), 0 (neutral), and 1 (positive).

By aggregating these sentiment scores at a state level, we were able to create a map showing the overall sentiment in each state at different points during the pandemic. The sentiment scores were smoothed using a Gaussian kernel to reduce noise and emphasize general trends.

## Results

The resulting visualizations offer a unique perspective on the public response to the pandemic. They highlight geographic differences in sentiment and show how sentiment has changed over time.

![Sentiment of US over time](https://github.com/AntanasMurelis/SentPred/blob/fa4f0267ba3f7222a91532463b138f36ca541404/Movies/sentiment_map-13.gif)


## Conclusion

This project showcases the potential of NLP and sentiment analysis in understanding public opinion during a major event like the COVID-19 pandemic. It provides valuable insights that could be used by policymakers, researchers, and public health officials to understand public sentiment, inform decision making, and communicate more effectively.

Please refer to the notebooks and scripts for more detailed information on the data processing, model training, and visualization steps.
