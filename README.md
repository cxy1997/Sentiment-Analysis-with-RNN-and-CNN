# Sentiment-Analysis
## Project of SJTU-CS438 Internet-based Information Extraction Technologies

For a given comment on anything included in the train corpus, the model predicts its polarity (*positive*/*negative*) with an architecture that first converts the words into embedding vectors, then passes them through a neural network combining LSTM and convolutional layers to achieve better performance.

## Dataset

We collected and **annoted** a dataset containing **34639 Chinese comments** and **13385 English comments** on *movies*, *books*, *music* and *electronic devices*, which can be found in the [dataset](./dataset) folder.

## Precision

- Chinese: 87.47%

- English: 83.69%

- You may use your own word embedding and train the model with a new language

## Examples

- `The cake is a lie.` *Negative*

- `I'm lovin' it.` *Positive*

## Team members:

- [Xiangyu Chen](https://github.com/cxy1997)

- [Zelin Ye](https://github.com/shinshiner)

- [Minchao Lee](https://github.com/MarshalLeeeeee)

- [Zhiwen Qiang](https://github.com/QLightman)
