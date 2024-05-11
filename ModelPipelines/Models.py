from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    LinearSVC,
    OneVsRest,
    NaiveBayes,
    DecisionTreeClassifier,
    MultilayerPerceptronClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, min, max, stddev, mean
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os

TARGET_COL = "condition"
FEATURES_COL = "features"


def read_data(preprocess=None, data_preprocessed=False):
    # Create a Spark session
    spark = SparkSession.builder.appName("SVC").getOrCreate()

    # Read the data from the CSV file
    module_dir = os.path.dirname(__file__)
    if data_preprocessed:
        train_data_path = os.path.join(
            module_dir, '../dataset/train_preprocessed.csv')
        test_data_path = os.path.join(
            module_dir, '../dataset/test_preprocessed.csv')
    else:
        train_data_path = os.path.join(
            module_dir, '../dataset/train_combined.csv')
        test_data_path = os.path.join(
            module_dir, '../dataset/test_combined.csv')

    train_data = spark.read.csv(train_data_path, header=True, inferSchema=True)
    test_data = spark.read.csv(test_data_path, header=True, inferSchema=True)

    if preprocess:
        if preprocess not in ["normalize", "standardize"]:
            raise ValueError(
                "preprocess should be either 'normalize' or 'standardize'")
        train_data, test_data = preprocess_data(
            train_data, test_data, preprocess)

    return train_data, test_data


def preprocess_data(train_data, test_data, transform="normalize"):
    features = train_data.columns
    features.remove(TARGET_COL)
    if transform == "standardize":
        # standardize the data
        for feature in features:
            mean_val = train_data.select(mean(col(feature))).collect()[0][0]
            std_val = train_data.select(stddev(col(feature))).collect()[0][0]
            train_data = train_data.withColumn(
                feature, (col(feature) - mean_val) / std_val)

            mean_val = test_data.select(mean(col(feature))).collect()[0][0]
            std_val = test_data.select(stddev(col(feature))).collect()[0][0]
            test_data = test_data.withColumn(
                feature, (col(feature) - mean_val) / std_val)

    elif transform == "normalize":
        # normalize the data
        for feature in features:
            min_val = train_data.select(min(col(feature))).collect()[0][0]
            max_val = train_data.select(max(col(feature))).collect()[0][0]
            train_data = train_data.withColumn(
                feature, (col(feature) - min_val) / (max_val - min_val))

            min_val = test_data.select(min(col(feature))).collect()[0][0]
            max_val = test_data.select(max(col(feature))).collect()[0][0]
            test_data = test_data.withColumn(
                feature, (col(feature) - min_val) / (max_val - min_val))

    return train_data, test_data


def data_formatting(train_data, test_data):
    # Create a vector assembler object
    features = train_data.columns
    features.remove(TARGET_COL)
    assembler = VectorAssembler(inputCols=features, outputCol=FEATURES_COL)

    # Transform the training and test data
    train_data = assembler.transform(train_data)
    test_data = assembler.transform(test_data)

    return train_data.select(FEATURES_COL, TARGET_COL), test_data.select(FEATURES_COL, TARGET_COL)


def lr_model(train_data, test_data):

    lr = LogisticRegression(maxIter=10, elasticNetParam=0.8,
                            family="multinomial",
                            featuresCol=FEATURES_COL, labelCol=TARGET_COL)

    # Train the model using the training sets
    lr_model = lr.fit(train_data)

    trainingSummary = lr_model.summary
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

    # Make predictions on the training and test data
    test_predictions = lr_model.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol=TARGET_COL)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Test Accuracy: {test_accuracy}")

    return lr_model


def ovr_model(train_data, test_data):

    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    ovr = OneVsRest(classifier=lsvc, labelCol=TARGET_COL)

    # Train the model using the training sets
    ovr_model = ovr.fit(train_data)

    # Make predictions on the training and test data
    test_predictions = ovr_model.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol=TARGET_COL)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Test Accuracy: {test_accuracy}")

    return ovr_model


def naive_model(train_data, test_data):

    nb = NaiveBayes(smoothing=1.0, modelType="multinomial",
                    labelCol=TARGET_COL, featuresCol=FEATURES_COL)

    # Train the model using the training sets
    nb_model = nb.fit(train_data)

    # Make predictions on the training and test data
    test_predictions = nb_model.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol=TARGET_COL)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Test Accuracy: {test_accuracy}")

    return nb_model


def decision_tree_model(train_data, test_data):

    dt = DecisionTreeClassifier(labelCol=TARGET_COL, featuresCol=FEATURES_COL)

    # Train the model using the training sets
    dt_model = dt.fit(train_data)

    # Make predictions on the training and test data
    test_predictions = dt_model.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol=TARGET_COL)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Test Accuracy: {test_accuracy}")

    return dt_model


def mlp_model(train_data, test_data):

    # Create a Multi-Layer Perceptron model
    layers = [train_data.select(
        FEATURES_COL).head().features.size, 100, 100, 3]
    mlp = MultilayerPerceptronClassifier(
        layers=layers, featuresCol=FEATURES_COL, labelCol=TARGET_COL)

    # Train the model using the training sets
    mlp_model = mlp.fit(train_data)

    # Make predictions on the training and test data
    test_predictions = mlp_model.transform(test_data)

    # evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(labelCol=TARGET_COL)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Test Accuracy: {test_accuracy}")


def Grid_Search(train_data, test_data, model_type="lr"):

    # Create an SVM model
    model = None
    paramGrid = None

    if model_type == "lr":
        model = LogisticRegression(
            featuresCol=FEATURES_COL, labelCol=TARGET_COL, family="multinomial")

        paramGrid = ParamGridBuilder() \
            .addGrid(model.elasticNetParam, [0, 0.5, 1]) \
            .addGrid(model.maxIter, [10, 100]) \
            .build()

    elif model_type == "lsvc":
        lsvc = LinearSVC()
        model = OneVsRest(classifier=lsvc, labelCol=TARGET_COL)

        paramGrid = ParamGridBuilder() \
            .addGrid(lsvc.regParam, [0.1, 0.01]) \
            .addGrid(lsvc.maxIter, [10, 100]) \
            .addGrid(lsvc.tol, [1e-3, 1e-4]) \
            .build()

    elif model_type == "nb":
        model = NaiveBayes(featuresCol=FEATURES_COL,
                           labelCol=TARGET_COL, modelType="multinomial")

        paramGrid = ParamGridBuilder() \
            .addGrid(model.smoothing, [0, 1, 2]) \
            .build()

    elif model_type == "dt":
        model = DecisionTreeClassifier(
            featuresCol=FEATURES_COL, labelCol=TARGET_COL)

        paramGrid = ParamGridBuilder() \
            .addGrid(model.maxDepth, [5, 10, 20]) \
            .addGrid(model.maxBins, [32, 64, 128]) \
            .build()

    else:
        raise ValueError(
            "model_type should be either 'lr', 'lsvc', 'nb' or 'dt'")

    # Create a binary classification evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol=TARGET_COL)

    # Create a cross-validator
    cv = CrossValidator(estimator=model,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)

    # Fit the cross-validator to the training data
    cvModel = cv.fit(train_data)

    # Get the best model
    bestModel = cvModel.bestModel

    # Print the best hyperparameters
    print("Best hyperparameters:")
    if model_type == "lr":
        print(f"elasticNetParam: {bestModel.getElasticNetParam()}")
        print(f"maxIter: {bestModel.getMaxIter()}")
    elif model_type == "lsvc":
        print(f"regParam: {bestModel.getClassifier().getRegParam()}")
        print(f"maxIter: {bestModel.getClassifier().getMaxIter()}")
        print(f"tol: {bestModel.getClassifier().getTol()}")
    elif model_type == "nb":
        print(f"smoothing: {bestModel.getSmoothing()}")
        print(f"modelType: {bestModel.getModelType()}")
    elif model_type == "dt":
        print(f"maxDepth: {bestModel.getMaxDepth()}")
        print(f"maxBins: {bestModel.getMaxBins()}")

    # Make predictions on the training and test data
    train_predictions = bestModel.transform(train_data)
    test_predictions = bestModel.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(
        labelCol=TARGET_COL, metricName="accuracy")
    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    return bestModel
