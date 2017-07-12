from nose.tools import assert_equal
import findspark
findspark.init()
from pyspark.sql import SparkSession, Row
from pyspark.ml import feature, classification
from pyspark.sql import types
import sparkml_pipes

sparkml_pipes.patch()

SPARK_SESSION = SparkSession.builder.getOrCreate()
SPARK_SESSION.sparkContext.setLogLevel("OFF")


def test_simplepipe():
    df = SPARK_SESSION.sparkContext.\
        parallelize([['this is a test'], ['this is another test']]).\
        toDF(schema=types.StructType().add('sentence', types.StringType()))

    pl = feature.Tokenizer().setInputCol('sentence') | \
        feature.CountVectorizer() | \
        feature.IDF()
    pl_model = pl.fit(df)
    pl_model.transform(df).count()


def test_unigram_and_bigram():
    df = SPARK_SESSION.sparkContext. \
        parallelize([['this is the best sentence ever'],
                     ['this is however the worst sentence available']]). \
        toDF(schema=types.StructType().add('sentence', types.StringType()))
    import requests
    stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

    tokenizer = feature.Tokenizer().setInputCol('sentence') | feature.StopWordsRemover(stopWords=stop_words)
    unigram = feature.CountVectorizer()
    bigram = feature.NGram() | feature.CountVectorizer()
    trigram = feature.NGram(n=3) | feature.CountVectorizer()
    tf = tokenizer | (unigram, bigram, trigram) | feature.VectorAssembler()
    tfidf = tf | feature.IDF().setOutputCol('features')

    tfidf_model = tfidf.fit(df)
    assert_equal(tfidf_model.transform(df).select('sentence', 'features').count(), 2)


def test_ml_pipe():
    df = SPARK_SESSION.sparkContext. \
        parallelize([Row(sentence='this is a test', label=0.),
                     Row(sentence='this is another test', label=1.)]).\
        toDF()

    pl = feature.Tokenizer().setInputCol('sentence') | feature.CountVectorizer()
    ml = pl | classification.LogisticRegression()

    ml_model = ml.fit(df)
    assert_equal(ml_model.transform(df).count(), 2)


def test_stackedml_pipe():
    df = SPARK_SESSION.sparkContext. \
        parallelize([Row(sentence='this is a test', label=0.),
                     Row(sentence='this is another test', label=1.)]).\
        toDF()

    pl = feature.Tokenizer().setInputCol('sentence') | feature.CountVectorizer()
    ml = pl | (classification.LogisticRegression(),) | feature.VectorAssembler() | \
        classification.\
        RandomForestClassifier().\
        setPredictionCol('final_prediction').\
        setRawPredictionCol('final_raw_predictions').\
        setProbabilityCol('final_probability')

    ml_model = ml.fit(df)
    assert_equal(ml_model.transform(df).count(), 2)
