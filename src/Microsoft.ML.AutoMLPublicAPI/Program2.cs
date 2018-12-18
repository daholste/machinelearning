using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class Program2
    {
        public static void Main2(string[] args)
        {
            const string trainDataPath = @"C:\temp\ntrn.csv";
            const string testDataPath = @"C:\temp\ntst.csv";

            var mlContext = new MLContext();
            TextLoader textLoader = new TextLoader(mlContext,
                new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("VendorId", DataKind.Text, 0),
                        new TextLoader.Column("PaymentType", DataKind.Text, 1),
                        new TextLoader.Column("TripDistance", DataKind.R4, 2),
                        new TextLoader.Column("TripTime", DataKind.R4, 3),
                        new TextLoader.Column("PassengerCount", DataKind.R4, 4),
                        new TextLoader.Column("RateCode", DataKind.Text, 5),
                        new TextLoader.Column("Label", DataKind.R4, 6)
                    }
                });

            // load data
            var trainData = textLoader.Read(trainDataPath);
            var testData = textLoader.Read(testDataPath);

            var columnPurposes = new Dictionary<string, ColumnPurpose>()
            {
                { "VendorId", ColumnPurpose.Categorical },
                { "PaymentType", ColumnPurpose.Categorical},
                { "TripDistance", ColumnPurpose.Numerical},
                { "TripTime", ColumnPurpose.Numerical},
                { "PassengerCount", ColumnPurpose.Numerical},
                { "RateCode", ColumnPurpose.Categorical },
                { "Label", ColumnPurpose.Label},
            };

            // custom preprocessor
            var preprocessor = mlContext.Transforms.CopyColumns("PaymentType", "PaymentTypeCopy");
            var validationData = preprocessor.Fit(testData).Transform(testData);

            var amlConfig = new AutoMlConfig
            {
                ValidationData = validationData,
                TrainingData = trainData,
                MaxIterationCount = 3,
                ColumnPurposes = columnPurposes
            };

            var experiment = new AutoMlExperiment(amlConfig);

            experiment.AddIterationObserver(new IterationProgressMonitor());

            var result = experiment.Fit(mlContext);

            Console.WriteLine($"Ran {result.IterationResults.Count} iterations");

            // run AutoML on test data
            var transformedOutput = result.BestIterationModel.Transform(testData);
            var results = mlContext.Regression.Evaluate(transformedOutput);

            Console.WriteLine($"R^2: {results.RSquared}");

            Console.WriteLine($"R^2 from GetResult: {result.BestIteration.Score.RSquared}");
        }
    }
}
