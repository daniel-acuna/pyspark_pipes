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
        raise Exception("For now, it only accepts primitive estimators and transfomers " \
                        "to the right-hand side")

    # At the start of the Pipeline we need to put together two things
    if not isinstance(other, Pipeline):
        return Pipeline().setStages([other]) | self
    else:
        last_step = other.getStages()[-1]
        if isinstance(last_step, HasOutputCol):
            # let's generate some random string to represent the column's name
            last_step_output = last_step.getOutputCol()
        if isinstance(last_step, HasPredictionCol):
            last_step_output = last_step.getPredictionCol()

        # should we connect input with output?
        if isinstance(self, HasInputCol):
            if not self.isSet('inputCol'):
                self.setInputCol(last_step_output)

        return Pipeline().setStages(other.getStages() + [self])

Params.__ror__ = pipe_function
