using Microsoft.ML.Legacy.Models;
using Microsoft.ML.PipelineInference;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.PipelineInference;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using static Microsoft.ML.Runtime.Data.IO.TextSaver;
using static Microsoft.ML.Runtime.PipelineInference.AutoInference;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Program
    {
        private static string _trainDataPath = $"/data/train.csv";
        private static string _validDataPath = $"/data/valid.csv";
        private static string _testDataPath = $"/data/test.csv";
        private static string _trainValidDataPath = $"/data/train_valid.csv";
        private static string _labelColName = "Label";

        public static void Main(string[] args)
        {
            var stopwatch = Stopwatch.StartNew();

            //var datasetName = "Physics";
            //var labelColName = "signal";
            //var datasetsDir = @"D:\RealSplitDatasets\";
            var numIterations = 200;
            var trainerKind = MacroUtils.TrainerKinds.SignatureRegressorTrainer;
            var metricToOptimize = PipelineSweeperSupportedMetrics.Metrics.RSquared;

            MyGlobals.OutputDir = args[0];
            MyGlobals.Stopwatch = stopwatch;

            var dir = AppDomain.CurrentDomain.BaseDirectory;
            using (var env = new ConsoleEnvironment())
            using (AssemblyLoadingUtils.CreateAssemblyRegistrar(env, dir))
            {
                var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(env,
                        _trainDataPath, _labelColName);

                var trainData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, _trainDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;
                var validData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, _validDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;
                var testData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, _testDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;

                MergeTrainValidateDatasets();
                var trainValidateData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, _trainValidDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;

                var metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(metricToOptimize);
                var rocketEngine = new RocketEngine(env, new RocketEngine.Arguments() { });
                var terminator = new IterationTerminator(numIterations);

                AutoMlMlState amls = new AutoMlMlState(env, metric, rocketEngine, terminator, trainerKind,
                    trainData, validData);
                var bestPipeline = amls.InferPipelines(1, 3, 100);

                bestPipeline.RunTrainTestExperiment(trainData,
                    testData, metric, trainerKind, out var testMetricVal, out var trainMetricVal);

                Ensemble(env, trainData, testData);

                File.AppendAllText($"{MyGlobals.OutputDir}/test_metric.txt", $"{testMetricVal}\r\n");
            }

            File.AppendAllText($"{MyGlobals.OutputDir}/time.txt", $"{stopwatch.ElapsedMilliseconds}ms\r\n");
        }

        private static void MergeTrainValidateDatasets()
        {
            var trainLines = File.ReadAllLines(_trainDataPath);
            var validLines = File.ReadAllLines(_validDataPath);

            File.Delete(_trainValidDataPath);
            File.WriteAllLines(_trainValidDataPath, trainLines);
            File.AppendAllLines(_trainValidDataPath, validLines.Skip(1));
        }

        private static void Ensemble(IHostEnvironment env, IDataView trainData, IDataView testData)
        {
            var ch = env.Start("hi");
            //var numEnsembles = MyGlobals.BestModels.Count() / 10;
            var numEnsembles = 20;
            var topModels = MyGlobals.BestModels.Take(numEnsembles).Select(m => m.Value);
            var mergedTrainData = CreateMergedData(env, trainData, _trainDataPath, "trainMerged", topModels);
            var ctx = new RegressionContext(env);
            var trainer = ctx.Trainers.StochasticDualCoordinateAscent();
            TrainUtils.AddNormalizerIfNeeded(env, ch, trainer, ref mergedTrainData, "Features", Data.NormalizeOption.Auto);
            var model = trainer.Fit(mergedTrainData);
            var mergedTestData = CreateMergedData(env, testData, _testDataPath, "testMerged", topModels);
            mergedTestData = ApplyTransformUtils.ApplyAllTransformsToData(env, mergedTrainData, mergedTestData);
            var transformed = model.Transform(mergedTestData);
            var metrics = ctx.Evaluate(transformed);
            File.AppendAllText("ensembled", metrics.RSquared.ToString() + "\r\n");

            /*var testLines = File.ReadAllLines(_testDataPath).ToList();
            var labelIdx = testLines[0].Split(',').ToList().FindIndex(c => c == "Label");
            testLines = testLines.Skip(1).ToList();
            var scores = GetColumnValues(transformed, "Score");
            File.Delete("newFile");
            using(var newFile = new StreamWriter(File.OpenWrite("newFile")))
            {
                newFile.WriteLine("Label,Score");
                for(var i = 0; i < testLines.Count(); i++)
                {
                    newFile.WriteLine($"{testLines[i].Split(',')[labelIdx]},{Math.Pow(double.Parse(scores[i]), 3)}");
                }
            }

            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(env,
                        "newFile", _labelColName);
            var okayData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
            {
                InputFile = new SimpleFileHandle(env, "newFile", false, false),
                Arguments = textLoaderArgs
            }).Data;
            metrics = ctx.Evaluate(okayData);*/
        }

        private static IDataView CreateMergedData(IHostEnvironment env, IDataView data,
            string dataFilePath, string tmpFilePath, IEnumerable<IPredictorModel> topModels)
        {
            var scoresForTopModels = new List<IList<string>>();
            foreach (var topModel in topModels)
            {
                var scores = Score(env, data, topModel, MacroUtils.TrainerKinds.SignatureRegressorTrainer);
                scoresForTopModels.Add(scores);
            }
            File.Delete(tmpFilePath);
            var dataLines = File.ReadAllLines(dataFilePath);
            using (var tmpFile = new StreamWriter(File.OpenWrite(tmpFilePath)))
            {
                var headers = dataLines[0];
                //var headersSb = new StringBuilder(headers);
                var headersSb = new StringBuilder("Label");
                for (var i = 0; i < scoresForTopModels.Count(); i++)
                {
                    headersSb.Append(",");
                    headersSb.Append("fsalklkfsjdk");
                    headersSb.Append(i);
                }
                tmpFile.WriteLine(headersSb.ToString());

                var labelIdx = headers.Split(',').ToList().FindIndex(x => x=="Label");

                for (var i = 1; i < dataLines.Count(); i++)
                {
                    var sb = new StringBuilder();

                    var dataLine = dataLines[i];
                    //sb.Append(dataLine);
                    sb.Append(dataLine.Split(",")[labelIdx]);
                    for (var j = 0; j < scoresForTopModels.Count(); j++)
                    {
                        sb.Append(",");
                        sb.Append(scoresForTopModels[j][i-1]);
                    }
                    tmpFile.WriteLine(sb.ToString());
                }
            }

            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(env,
                        tmpFilePath, _labelColName);
            var mergedTrainData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
            {
                InputFile = new SimpleFileHandle(env, tmpFilePath, false, false),
                Arguments = textLoaderArgs
            }).Data;

            return mergedTrainData;
        }

        private static string CubeRoot(string s)
        {
            double d = double.Parse(s);
            return Math.Cbrt(d).ToString();
        }

        private static IList<string> Score(IHostEnvironment env, IDataView data, IPredictorModel model,
            MacroUtils.TrainerKinds trainerKind)
        {
            var experiment = env.CreateExperiment();
            var scoreInput = new Legacy.Transforms.DatasetScorer
            {
                Data = new Var<IDataView>() { VarName = "data" },
                PredictorModel = new Var<IPredictorModel> { VarName = "model" },
            };
            var scoreOutput = experiment.Add(scoreInput);
            var evalInput = new Legacy.Models.RegressionEvaluator
            {
                Data = scoreOutput.ScoredData
            };
            var evalOutput = experiment.Add(evalInput);
            experiment.Compile();

            experiment.SetInput("data", data);
            experiment.SetInput("model", model);
            experiment.Run();

            var scoredData = experiment.GetOutput(evalOutput.PerInstanceMetrics);
            var scores = GetColumnValues(scoredData, "Score");
            return scores;
            /*if (trainerKind == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
            {
                trainTestOutput.BinaryClassificationMetrics = BinaryClassificationMetrics.FromMetrics(
                    environment,
                    experiment.GetOutput(trainTestNodeOutput.OverallMetrics),
                    experiment.GetOutput(trainTestNodeOutput.ConfusionMatrix)).FirstOrDefault();
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer)
            {
                trainTestOutput.ClassificationMetrics = ClassificationMetrics.FromMetrics(
                    environment,
                    experiment.GetOutput(trainTestNodeOutput.OverallMetrics),
                    experiment.GetOutput(trainTestNodeOutput.ConfusionMatrix)).FirstOrDefault();
            }
            else if (trainerKind == MacroUtils.TrainerKinds.SignatureRegressorTrainer)
            {
                trainTestOutput.RegressionMetrics = RegressionMetrics.FromOverallMetrics(
                    environment,
                    experiment.GetOutput(trainTestNodeOutput.OverallMetrics)).FirstOrDefault();
            }
            else if (Kind == MacroUtilsTrainerKinds.SignatureClusteringTrainer)
            {
                trainTestOutput.ClusterMetrics = ClusterMetrics.FromOverallMetrics(
                    environment,
                    experiment.GetOutput(trainTestNodeOutput.OverallMetrics)).FirstOrDefault();
            }

            var metricsRaw = experiment.GetOutput(evalOutput.OverallMetrics);
            var metrics = RegressionMetrics.FromOverallMetrics(env, metricsRaw);
             */
        }

        private static IList<string> GetColumnValues(IDataView data, string colName)
        {
            var values = new List<string>();
            var cursor = data.GetRowCursor((i) => true);
            var scoreColIdx = cursor.Schema.GetColumns().First(c => c.column.Name == colName).index;
            var valueWriter = ValueWriter.Create(cursor, scoreColIdx, ' ');

            Action<StringBuilder, int> append = (sb, i) => {
                values.Add(sb.ToString());
            };
            while (cursor.MoveNext())
            {
                valueWriter.WriteData(append, out var len);
            }

            return values;
        }
    }
}
