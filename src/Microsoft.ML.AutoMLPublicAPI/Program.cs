using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string testDataPath = @"C:\data\test.csv";

            var mlContext = new MLContext();
            TextLoader textLoader = new TextLoader(mlContext,
                new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("VendorId", DataKind.Text, 0),
                        new TextLoader.Column("RateCode", DataKind.Text, 1),
                        new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                        new TextLoader.Column("TripTime", DataKind.R4, 3),
                        new TextLoader.Column("TripDistance", DataKind.R4, 4),
                        new TextLoader.Column("PaymentType", DataKind.Text, 5),
                        new TextLoader.Column("FareAmount", DataKind.R4, 6)
                    }
                });

            // load data
            var trainData = textLoader.Read(trainDataPath);
            var testData = textLoader.Read(testDataPath);

            var columnPurposes = new Dictionary<string, ColumnPurpose>()
            {
                { "VendorId", ColumnPurpose.Categorical },
                { "RateCode", ColumnPurpose.Categorical },
                { "PassengerCount", ColumnPurpose.Numerical},
                { "TripTime", ColumnPurpose.Numerical},
                { "TripDistance", ColumnPurpose.Numerical},
                { "PaymentType", ColumnPurpose.Categorical},
                { "FareAmount", ColumnPurpose.Label},
            };

            // run AutoML
            var transform = mlContext.Transforms.CopyColumns("PaymentType", "PaymentType1");
            var autoMlTrainer = mlContext.Regression.Trainers.Auto(columnPurposes, maxIterations: 10);
            var pipeline = transform.Append(autoMlTrainer);
            var model = pipeline.Fit(trainData);

            // apply AutoML results
            var transformedOutput = model.Transform(testData);
            var results = mlContext.Regression.Evaluate(transformedOutput);
        }
    }
}
