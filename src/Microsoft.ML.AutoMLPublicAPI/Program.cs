using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            const string trainDataPath = @"C:\data\sample_train.csv";
            const string testDataPath = @"C:\data\sample_test.csv";

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
                        new TextLoader.Column("Label", DataKind.R4, 6)
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
                { "Label", ColumnPurpose.Label},
            };

            // custom preprocessor
            var preprocessor = mlContext.Transforms.CopyColumns("PaymentType", "PaymentTypeCopy");
            var validationData = preprocessor.Fit(testData).Transform(testData);

            var autoMlTrainer = mlContext.Regression.Trainers.Auto(columnPurposes, maxIterations: 3, validationData: validationData);
            var pipeline = preprocessor.Append(autoMlTrainer);
            var model = pipeline.Fit(trainData);

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.Regression.Evaluate(transformedOutput);
            Console.WriteLine($"R^2: {results.RSquared}");

            var predictionEngine = model.MakePredictionFunction<Trip, TripPrice>(mlContext);
            var prediction = predictionEngine.Predict(new Trip()
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1200,
                TripDistance = 1,
                PaymentType = "CRD"
            });
            Console.WriteLine($"Prediction score: {prediction.Score}");

            string modelPath = $"Model.zip";
            using (var fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            ITransformer savedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                savedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine related to the loaded trained model
            var predFunction = savedModel.MakePredictionFunction<Trip, TripPrice>(mlContext);
            prediction = predictionEngine.Predict(new Trip()
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1200,
                TripDistance = 1,
                PaymentType = "CRD"
            });
            Console.WriteLine($"Prediction score: {prediction.Score}");
        }

        public class Trip
        {
            public string VendorId;
            public string RateCode;
            public float PassengerCount;
            public float TripTime;
            public float TripDistance;
            public string PaymentType;
            public float Label;
        }

        public class TripPrice
        {
            public float Score;
        }
    }
}
