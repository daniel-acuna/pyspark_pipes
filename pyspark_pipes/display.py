"""
Some pipeline display functionality
"""

from pyspark.ml import Pipeline
from pyspark.ml.param.shared import HasFeaturesCol, HasInputCol, \
    HasInputCols, HasPredictionCol, HasOutputCol


def print_stage(p):
    if isinstance(p, Pipeline):
        return "[\n" + ','.join([print_stage(s) for s in p.getStages()]) + "\n]"
    else:
        r = ""
        if isinstance(p, HasInputCol):
            r += p.getInputCol()
        elif isinstance(p, HasInputCols):
            r += str(p.getInputCols())
        elif isinstance(p, HasFeaturesCol):
            r += p.getFeaturesCol()

        r += " - "
        if isinstance(p, HasOutputCol):
            r += p.getOutputCol()
        elif isinstance(p, HasPredictionCol):
            r += p.getPredictionCol()
        return r
