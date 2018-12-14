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
            const string trainDataPath = @"C:\data\sample_train2.csv";
            const string validationDataPath = @"C:\data\sample_valid2.csv";
            const string testDataPath = @"C:\data\sample_test2.csv";
            const string labelColName = "Label";

            var mlContext = new MLContext();

            // load data
            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(mlContext, trainDataPath, labelColName);
            var textLoader = new TextLoader(mlContext, textLoaderArgs);
            var trainData = textLoader.Read(trainDataPath);
            var validationData = textLoader.Read(validationDataPath);
            var testData = textLoader.Read(testDataPath);

            var sdca = mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent();
            var sdcaModel = sdca.Fit(trainData);
            var sdcaScoredData = sdcaModel.Transform(testData);
            var sdcaResults = mlContext.BinaryClassification.Evaluate(sdcaScoredData);
            Console.WriteLine($"\r\nSDCA Accuracy: {sdcaResults.Accuracy}");

            var lightGbm = mlContext.BinaryClassification.Trainers.LightGbm();
            var lightGbmModel = lightGbm.Fit(trainData);
            var lightGbmScoredData = lightGbmModel.Transform(testData);
            var lightGbmResults = mlContext.BinaryClassification.Evaluate(lightGbmScoredData);
            Console.WriteLine($"\r\nLightGBM Accuracy: {lightGbmResults.Accuracy}");

            // custom preprocessor
            var autoMlTrainer = mlContext.BinaryClassification.Trainers.Auto(maxIterations: 25, validationData: validationData);
            var model = autoMlTrainer.Fit(trainData);

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"\r\nAuto Accuracy: {results.Accuracy}");

            Console.ReadLine();
        }
    }
}
