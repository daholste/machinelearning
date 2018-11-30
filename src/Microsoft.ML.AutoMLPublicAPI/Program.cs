using Microsoft.ML.Runtime.Data;
using System;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string validationDataPath = @"C:\data\valid.csv";
            const string testDataPath = @"C:\data\test.csv";

            // infer columns
            var mlContext = new MLContext();
            var columns = mlContext.InferColumns(trainDataPath, label: "y");

            // load data
            var trainData = mlContext.InferDataView(trainDataPath, columns);
            var validationData = mlContext.InferDataView(validationDataPath, columns);
            var testData = mlContext.InferDataView(testDataPath, columns);

            // optionally, infer subcontext
            var inferredContext = mlContext.InferTrainContext(trainData);

            // manually select subcontext
            var autoMlContext = mlContext.Regression.AutoMl;

            // run AutoML
            var result = autoMlContext.InferPipeline(columns, trainData, validationData, testData, maxIterations: 10);

            //// can leverage Auto ML results to:

            // (1) examine validation set results,
            Console.WriteLine(result.ValidationResult.RSquared);

            // (2) re-use trained model to predict test data, or
            var trainedModel = result.TrainedModel;
            var testDataResults = trainedModel.Transform(testData);

            // (3) re-train winnig pipeline on new data,
            var newTrainedModel = result.Estimator.Fit(validationData);
        }
    }
}
