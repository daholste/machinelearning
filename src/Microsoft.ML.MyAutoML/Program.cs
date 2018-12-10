using Microsoft.ML.Legacy.Models;
using Microsoft.ML.PipelineInference;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Transforms;
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

        private const int NumIterations = 200;
        private const int EnsembleSize = 10;
        private const int NumAutoMlItersForEnsembledDataset = 50;

        public static void MainSafe(string[] args)
        {
            // get trainer kind / problem type
            MacroUtils.TrainerKinds trainerKind;
            switch (args[1])
            {
                case "regression":
                    trainerKind = MacroUtils.TrainerKinds.SignatureRegressorTrainer;
                    break;
                case "binaryclassification":
                    trainerKind = MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer;
                    break;
                case "multiclassclassification":
                    trainerKind = MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer;
                    break;
                default:
                    throw new Exception("unsupported problem type");
            }

            // get metric to optimize
            var metricToOptimize = PipelineSweeperSupportedMetrics.Metrics.Accuracy;
            if (trainerKind == MacroUtils.TrainerKinds.SignatureRegressorTrainer)
            {
                metricToOptimize = PipelineSweeperSupportedMetrics.Metrics.RSquared;
            }

            var stopwatch = Stopwatch.StartNew();

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
                var terminator = new IterationTerminator(NumIterations);

                AutoMlMlState amls = new AutoMlMlState(env, metric, rocketEngine, terminator, trainerKind,
                    trainData, validData);
                var bestPipelines = amls.InferPipelines(1, 3, 100);
                var bestPipeline = bestPipelines.First();

                bestPipeline.RunTrainTestExperiment(trainData,
                    testData, metric, trainerKind, out var testMetricVal, out var trainMetricVal);

                Ensemble(env, trainData, validData, testData, bestPipelines, metric, trainerKind);

                File.AppendAllText($"{MyGlobals.OutputDir}/test_metric.txt", $"{testMetricVal}\r\n");
            }

            File.AppendAllText($"{MyGlobals.OutputDir}/time.txt", $"{stopwatch.ElapsedMilliseconds}ms\r\n");
        }

        public static void Main(string[] args)
        {
            try
            {
                MainSafe(args);
            }
            catch(Exception ex)
            {
                System.Console.WriteLine($"Fatal exception: {ex}");
            }
        }

        private static void MergeTrainValidateDatasets()
        {
            var trainLines = File.ReadAllLines(_trainDataPath);
            var validLines = File.ReadAllLines(_validDataPath);

            File.Delete(_trainValidDataPath);
            File.WriteAllLines(_trainValidDataPath, trainLines);
            File.AppendAllLines(_trainValidDataPath, validLines.Skip(1));
        }

        private static void Ensemble(IHostEnvironment env, IDataView trainData, IDataView validData, IDataView testData,
            IEnumerable<PipelinePattern> topPipelines, SupportedMetric metric, MacroUtils.TrainerKinds trainerKind)
        {
            var topTrainedModels = MyGlobals.BestModels.Select(m => m.Value);

            var scoredValidData = CreateMergedData(env, validData, "validMerged", topTrainedModels);
            var scoredTestData = CreateMergedData(env, testData, "testMerged", topTrainedModels);

            /*
            var ch = env.Start("hi");
            var numFolds = 5;

            var splitOutput = CVSplit.Split(env, new CVSplit.Input { Data = trainData, NumFolds = numFolds });
            var trainMergedFilePath = "trainMerged";
            var validMergedFilePath = "validMerged";
            var testMergedFilePath = "testMerged";
            var splitScoredTrainData = new List<IDataView>();

            for(var i = 0; i < numFolds; i++)
            {
                System.Console.WriteLine($"Starting ensembling fold {i}");
                var splitTrainData = splitOutput.TrainData[i];
                var splitTestData = splitOutput.TestData[i];

                var trainedModels = new List<IPredictorModel>();
                for (var  j = 0; j < EnsembleSize; j++)
                {
                    System.Console.WriteLine($"Ensembling fold {i}, training model {j}");
                    var topPipeline = topPipelines.ElementAt(j);
                    var trainedModel = topPipeline.RunTrainTestExperiment(splitTrainData, splitTestData, metric, trainerKind,
                        out var _1, out var _2);
                    trainedModels.Add(trainedModel);
                }

                var scoredTrainDataFold = CreateMergedData(env, splitTestData, trainMergedFilePath + i, trainedModels);
                splitScoredTrainData.Add(scoredTrainDataFold);
            }

            WriteDataToFile(env, splitScoredTrainData, trainMergedFilePath);
            var scoredTrainData = LoadDataFromFile(env, trainMergedFilePath, _labelColName);
            var scoredValidData = CreateMergedData(env, validData, validMergedFilePath, topTrainedModels);
            var scoredTestData = CreateMergedData(env, testData, testMergedFilePath, topTrainedModels);

            var rocketEngine = new RocketEngine(env, new RocketEngine.Arguments() { });
            var terminator = new IterationTerminator(NumAutoMlItersForEnsembledDataset);
            AutoMlMlState amls = new AutoMlMlState(env, metric, rocketEngine, terminator, trainerKind,
                    scoredTrainData, scoredValidData);
            var bestPipelines = amls.InferPipelines(1, 3, 100);
            var bestPipeline = bestPipelines.First();
            bestPipeline.RunTrainTestExperiment(scoredTrainData,
                    scoredTestData, metric, trainerKind, out var testMetricVal, out var trainMetricVal);
            File.AppendAllText($"{MyGlobals.OutputDir}/ensembled", testMetricVal + "\r\n");*/

            /*var ctx = new RegressionContext(env);
            var trainer = ctx.Trainers.FastTree(numLeaves: 2, minDatapointsInLeaves: 1, numTrees: 200);
            TrainUtils.AddNormalizerIfNeeded(env, ch, trainer, ref scoredTrainData, "Features", Data.NormalizeOption.Auto);
            var model = trainer.Fit(scoredTrainData);
            scoredTestData = ApplyTransformUtils.ApplyAllTransformsToData(env, scoredTestData, scoredTestData);
            var transformed = model.Transform(scoredTestData);
            var metrics = ctx.Evaluate(transformed);
            File.AppendAllText($"{MyGlobals.OutputDir}/ensembled", metrics.RSquared.ToString() + "\r\n");

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
            string filePath, IEnumerable<IPredictorModel> topModels)
        {
            var scoresForTopModels = new List<IList<string>>();
            foreach (var topModel in topModels)
            {
                var scores = Score(env, data, topModel, MacroUtils.TrainerKinds.SignatureRegressorTrainer);
                scoresForTopModels.Add(scores);
            }

            var tmpFilePath = "tmpFile";
            WriteDataToFile(env, data, tmpFilePath);
            var dataLines = File.ReadAllLines(tmpFilePath);
            File.Delete(tmpFilePath);
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
            var tmpData = LoadDataFromFile(env, tmpFilePath, _labelColName);
            WriteDataToFile(env, tmpData, filePath);
            return LoadDataFromFile(env, filePath, _labelColName);
        }

        private static IDataView LoadDataFromFile(IHostEnvironment env, string filePath, string labelColName)
        {
            var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(env, filePath, _labelColName);
            var dataView = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
            {
                InputFile = new SimpleFileHandle(env, filePath, false, false),
                Arguments = textLoaderArgs
            }).Data;
            return dataView;
        }

        private static void WriteDataToFile(IHostEnvironment env, IDataView data, string path)
        {
            WriteDataToFile(env, new [] { data }, path);
        }

        private static void WriteDataToFile(IHostEnvironment env, IEnumerable<IDataView> dataViews, string path)
        {
            var firstDataView = dataViews.First();
            var cols = firstDataView.Schema.GetColumns();
            var orderedCols = cols.OrderBy(c => c.column.Name).Select(c => c.index);
            var numCols = orderedCols.Count();

            File.Delete(path);
            using (var streamWriter = new StreamWriter(path))
            {
                for (var i = 0; i < dataViews.Count(); i++)
                {
                    var data = dataViews.ElementAt(i);
                    var cursor = data.GetRowCursor((c) => true);
                    var pipes = new ValueWriter[numCols];
                    for (int j = 0; j < numCols; j++)
                    {
                        var col = orderedCols.ElementAt(j);
                        pipes[j] = ValueWriter.Create(cursor, col, ',');
                    }

                    if (i == 0)
                    {
                        var headersSb = new StringBuilder();
                        Action<StringBuilder, int> headerAppendFunc = (sb, n) => headersSb.Append("," + sb);
                        for (int k = 0; k < numCols; k++)
                        {
                            pipes[k].WriteHeader(headerAppendFunc, out var length);
                        }
                        streamWriter.WriteLine(headersSb.Remove(0, 1));
                    }

                    while (cursor.MoveNext())
                    {
                        var rowSb = new StringBuilder();
                        Action<StringBuilder, int> appendFunc = (sb, n) => rowSb.Append("," + sb);
                        for (int k = 0; k < numCols; k++)
                        {
                            pipes[k].WriteData(appendFunc, out var length);
                        }
                        streamWriter.WriteLine(rowSb.Remove(0, 1));
                    }
                }
            }
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
            var evalInput = new Legacy.Models.BinaryClassificationEvaluator
            {
                Data = scoreOutput.ScoredData
            };
            var evalOutput = experiment.Add(evalInput);
            experiment.Compile();

            experiment.SetInput("data", data);
            experiment.SetInput("model", model);
            experiment.Run();

            var scoredData = experiment.GetOutput(evalOutput.PerInstanceMetrics);
            var scores = GetColumnValues(scoredData, "Probability");
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
