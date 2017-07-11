from nose.tools import assert_true, assert_equal, assert_raises
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import feature
from pyspark.sql import types

from sparkml_pipes.ml_pipes import pipe_function
from sparkml_pipes.base_pipe import take

SPARK_SESSION = SparkSession.builder.getOrCreate()
SPARK_SESSION.sparkContext.setLogLevel("OFF")


def test_rddcreation():
    # create session
    # SPARK_SESSION = SparkSession.builder.getOrCreate()
    #
    # rdd = SPARK_SESSION.sparkContext.parallelize([1, 2, 3])
    # assert_equal(rdd.count(), 3)
    pass


def test_basicpipes():
    assert_equal(len(list([1, 2, 3] | take(2))), 2)


def test_otherpipes():

    df = SPARK_SESSION.sparkContext.\
        parallelize([['this is a test'], ['this is another test']]).\
        toDF(schema=types.StructType().add('sentence', types.StringType()))

    pl = feature.Tokenizer().setInputCol('sentence') \
         | (feature.CountVectorizer() \
         | feature.IDF())
    pl_model = pl.fit(df)
    pl_model.transform(df).show()

if __name__ == '__main__':
    test_otherpipes()
