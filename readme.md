# pyspark_pipes: write complex Spark ML pipelines with ease

## Introduction

Writing analytic pipelines is part of being a data scientist. 
Spark ML has [Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html) 
for working with Big Data. 
However, the syntax for creating, modifying, and maintaining complex pipes 
can be cumbersome. 

This packages tries to solve this problem by creating a *pipe* operator `|` 
that seamlessly stitches together instances of `Estimator` or `Transformer`. 
It automatically figures out how to match the output or 
prediction columns of previous steps
with the input column or columns of the next steps. 
It also allows to broadcast one output column to several other steps. Or 
vice versa take several output columns as an input to one step
(e.g., when using [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)).

Inspiration for writing the pipe operator is due to 
https://github.com/JulienPalard/Pipe

### Get started

**vanilla Spark ML**
```python
Pipeline(stages=[feature.Tokenizer().setInputColumn('sentence').setOutputCol('words'),
    feature.CountVectorizer().setInputCol('words').setOutputCol('tf'),
    classification.LogisticRegression().setFeaturesCol('tf')])
```

**with `pyspark_pipes`**
```python
import pyspark_pipes
pyspark_pipes.patch()

pl = feature.Tokenizer().setInputCol('sentence') | \
    feature.CountVectorizer() | \
    feature.LogisticRegression()
```

## Details

I started this package out of frustration while 
developing NLP analyses. These analyses
typically required several steps to produce features, such as tokenization, 
stop word removal, and count of term vectors. The problems started
when I wanted to add additional steps in the middle of the pipeline.

In vanilla Spark, an initial analysis might look like this:

```python
pl = Pipeline(stages=[
    feature.Tokenizer().setInputColumn('sentence').setOutputCol('words'),
    feature.CountVectorizer().setInputCol('words').setOutputCol('tf'),
    classification.LogisticRegression().setFeaturesCol('tf')
    ])
```

Now, this is  already verbose. But now suppose that we want to use the tf-idf vector instead of the raw tf vectors. 
Then, we would need to do something like this

```python
pl_idf = Pipeline(stages=[
    feature.Tokenizer().setInputColumn('sentence').setOutputCol('words'),
    feature.CountVectorizer().setInputCol('words').setOutputCol('tf'),
    feature.IDF().setInputCol('tf').setOutputCol('tfidf'),
    classification.LogisticRegression().setFeaturesCol('tfidf')])
```

This insertion changed the pipeline in two points: the point of insertion and subsequent stage.

Compare it with using `pyspark_pipes` for the tf pipeline

```python
pl = feature.Tokenizer().setInputCol('sentence') | \
    feature.CountVectorizer() | \
    classification.LogisticRegression()
```

and the tf-idf pipeline

```python
pl_idf = feature.Tokenizer().setInputCol('sentence') | \
    feature.CountVectorizer() | \
    feature.IDF() | \
    classification.LogisticRegression()
```

And there it is! It is much easier to read, modify, and maintain!
### Advanced examples

#### Bigrams

 ```python
 from pyspark.ml import feature, classification
# base tokenizer
tokenizer = feature.Tokenizer().setInputCol('sentence') | feature.StopWordsRemover(stopWords=stop_words)
# unigrams and bigrams
unigram = feature.CountVectorizer()
bigram = feature.NGram(n = 2) | feature.CountVectorizer()
# put together unigrams and bigrams
tf = tokenizer | (unigram, bigram) | feature.VectorAssembler()
pl_tfidf = tf | feature.IDF().setOutputCol('features')
```

#### Wisdom of the crowds - multiple algorithms

 ```python
from pyspark.ml import feature, classification

features = feature.Tokenizer().setInputCol('sentence') | \
    feature.CountVectorizer()
meta_features = (classification.LogisticRegression(),
        classification.RandomForestClassifier(),
        classification.LogisticRegression().setElasticNetParam(0.2),
        classification.GBTClassifier()
        ) | feature.VectorAssembler()
combiner = classification.LogisticRegression()
pl_woc = features | meta_features | combiner.setPredictionCol('prediction')
```


## How does it work?

`pyspark_pipes` "monkey patches" the `Params` class, which is the base for `Estimator` and `Transformer`. It understands
that we are trying to chain together these objects and makes the output column of previous steps coincide with
the input column of the next steps. At the moment, it is a very hacky solution to a very common problem. As
long as Python does not have _extensions_ for methods and attributes 
(such as [Kotlin](https://kotlinlang.org/docs/reference/extensions.html)), I do not see another solution.

Feel free to file an issue or contribute, or both!

## TODO

1. Saving the fitted pipelines has not been tested

For more, check the issues!

## Author

Daniel E. Acuña, http://acuna.io  
Assistant Professor  
[School of Information Studies](http://ischool.syr.edu)  
[Syracuse University](http://syracuse.edu)  

## License

Copyright (c) 2017, Daniel E. Acuña

[MIT License](https://opensource.org/licenses/MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
