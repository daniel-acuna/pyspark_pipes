"""
Monkey patch estimators and transformers (params) to support pipelining
"""

from pyspark.ml.param.shared import HasFeaturesCol, HasInputCol, \
    HasInputCols, HasLabelCol, HasPredictionCol, HasOutputCol, Params

from pyspark.ml import Pipeline

ALLOWED_TYPES = [HasFeaturesCol, HasInputCol, HasInputCols, HasLabelCol, HasPredictionCol, HasOutputCol]


def isanyinstance(o, typelist):
    return any([isinstance(o, t) for t in typelist])


def get_pipeline_laststep(p):
    if isinstance(p, Pipeline):
        return get_pipeline_laststep(p.getStages()[-1])
    else:
        return p


def get_pipeline_firststep(p):
    if isinstance(p, Pipeline):
        return get_pipeline_firststep(p.getStages()[0])
    else:
        return p


def left_pipe_function(self, other):

    # At the start of the Pipeline we need to put together two things
    if isanyinstance(other, ALLOWED_TYPES):
        result = Pipeline().setStages([other]) | self
    elif isinstance(other, Pipeline):
        last_step = get_pipeline_laststep(other)
        if isinstance(last_step, HasOutputCol):
            # let's generate some random string to represent the column's name
            last_step_output = last_step.getOutputCol()
        elif isinstance(last_step, HasPredictionCol):
            last_step_output = last_step.getPredictionCol()
        else:
            raise Exception("Type of step not supported")

        # should we connect input with output?
        first_step = get_pipeline_firststep(self)
        if isinstance(first_step, HasInputCol):
            if not first_step.isSet('inputCol'):
                first_step.setInputCol(last_step_output)
        if isinstance(first_step, HasFeaturesCol):
            if not first_step.isSet('featuresCol'):
                first_step.setFeaturesCol(last_step_output)

        result = Pipeline().setStages(other.getStages() + [self])
    elif isinstance(other, tuple) or isinstance(other, list):
        # check that connecting to one estimator or transformer
        if not isinstance(self, HasInputCols):
            raise Exception("When many to one connection, then receiver must accept multiple inputs")

        all_outputs = []
        all_objects = []
        for p in other:
            if not isinstance(p, NotBroadcasted):
                last_step = get_pipeline_laststep(p)

                if isinstance(last_step, HasOutputCol):
                    # let's generate some random string to represent the column's name
                    last_step_output = last_step.getOutputCol()
                elif isinstance(last_step, HasPredictionCol):
                    last_step_output = last_step.getPredictionCol()
                else:
                    raise Exception("It must contain output or predictoin")

                all_outputs.append(last_step_output)
                all_objects.append(p)
            else:
                all_objects.append(p.object)

        # should we connect input with output?
        first_step = get_pipeline_firststep(self)
        if not first_step.isSet('inputCols'):
            first_step.setInputCols(all_outputs)

        result = Pipeline().setStages(all_objects + [self])
    else:
        raise Exception("Type of pipeline not supported")
    return result


def right_pipe_function(self, other):
    if (isinstance(other, list) or isinstance(other, tuple)) and \
            (isanyinstance(self, ALLOWED_TYPES) or isinstance(self, Pipeline)):
        last_step = get_pipeline_laststep(self)

        if isinstance(last_step, HasOutputCol):
            # let's generate some random string to represent the column's name
            last_step_output = last_step.getOutputCol()
        elif isinstance(last_step, HasPredictionCol):
            last_step_output = last_step.getPredictionCol()
        else:
            raise Exception("It must contain output or prediction")

        for p in other:
            first_step = get_pipeline_firststep(p)

            if isinstance(first_step, HasInputCol):
                # let's generate some random string to represent the column's name
                if not first_step.isSet('inputCol'):
                    first_step.setInputCol(last_step_output)
            elif isinstance(first_step, HasFeaturesCol):
                if not first_step.isSet('featuresCol'):
                    first_step.setFeaturesCol(last_step_output)
            else:
                raise Exception("An step didn't allow inputs")

        result = [NotBroadcasted(self)] + list(other)
    else:
        result = left_pipe_function(other, self)
    return result


class NotBroadcasted:
    def __init__(self, params_object):
        self.object = params_object


def patch():
    Params.__or__ = right_pipe_function
    Params.__ror__ = left_pipe_function
