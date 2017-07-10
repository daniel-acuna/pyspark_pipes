from nose.tools import assert_true, assert_equal, assert_raises
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import feature


from sparkml_pipes.ml_pipes import pipe_function
from sparkml_pipes.base_pipe import take


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
    SPARK_SESSION = SparkSession.builder.getOrCreate()
    df = SPARK_SESSION.range(100)

    return df | feature.Tokenizer().setInputCol('id')