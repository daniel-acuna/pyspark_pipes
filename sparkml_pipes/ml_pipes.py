"""
Monkey patch estimators and transformers to support pipelining 
"""

from pyspark.ml.param.shared import HasFeaturesCol, HasInputCol, \
    HasInputCols, HasLabelCol, HasPredictionCol, HasOutputCol, Params

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
import uuid

ALLOWED_TYPES = [HasFeaturesCol, HasInputCol,
    HasInputCols, HasLabelCol, HasPredictionCol, HasOutputCol]

class DataFrameWrapper(object):
    def __init__(self, df):
        self.df = df


def isanyinstance(o, typelist):
    return any([isinstance(o, t) for t in typelist])


def pipe_function(self, other):
    if not isanyinstance(self, ALLOWED_TYPES):
        raise Exception("For now, it only accepts primitive estimators and transfomers")

    # check if input to pipeline is a dataframe
    if isinstance(other, DataFrame):
        if hasattr(other, 'df'):
            raise Exception("there is already a dataframe in the pipeline ({})".format(self.df))

        pl = Pipeline.setStages([self])
        pl.df = other
        return pl
    else:
        last_step = other.getStages()[-1]
        if isinstance(last_step, HasOutputCol):
            # let's generate some random string to represent the column's name
            if not last_step.isSet('OutputCol'):
                last_step_output = str(type(last_step)) + str(uuid.uuid1())[0:8]
            else:
                last_step_output = last_step.getOutputCol()
        if isinstance(last_step, HasPredictionCol):
            if not last_step.isSet('PredictionCol'):
                last_step_output = str(type(last_step)) + str(uuid.uuid1())[0:8]
            else:
                last_step_output = last_step.getPredictionCol()

        # should we connect input with output?
        if isinstance(self, HasInputCol):
            if not self.isSet('InputCol'):
                self.setInputCol(last_step_output)

        new_pl = Pipeline.setStages([other.getStages(), self])
        new_pl.df = other.df
        return new_pl

Params.__ror__ = pipe_function