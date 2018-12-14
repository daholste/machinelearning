using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using System;
using System.IO;

namespace Microsoft.ML.AutoMLPublicAPI2
{
    public class Program
    {
        public static void Main(string[] args)
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string validationDataPath = @"C:\data\valid.csv";
            const string testDataPath = @"C:\data\test.csv";
            const string labelColName = "Label";

            var mlContext = new MLContext();

            // load data
            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(mlContext, trainDataPath, labelColName);
            var textLoader = new TextLoader(mlContext, textLoaderArgs);
            var trainData = textLoader.Read(trainDataPath);
            var validationData = textLoader.Read(validationDataPath);
            var testData = textLoader.Read(testDataPath);

            // custom preprocessor
            var autoMlTrainer = mlContext.BinaryClassification.Trainers.Auto(maxIterations: 3, validationData: validationData);
            var model = autoMlTrainer.Fit(trainData);

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"Accuracy: {results.Accuracy}");
        }
    }
}
