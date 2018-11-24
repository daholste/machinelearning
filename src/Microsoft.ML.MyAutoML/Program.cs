using Microsoft.ML.PipelineInference;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using static Microsoft.ML.Runtime.PipelineInference.AutoInference;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Program
    {
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

            var dir = Directory.GetCurrentDirectory();
            using (var env = new ConsoleEnvironment())
            using (AssemblyLoadingUtils.CreateAssemblyRegistrar(env, dir))
            {
                string trainDataPath = $"/data/train.csv";
                string validDataPath = $"/data/valid.csv";
                string testDataPath = $"/data/test.csv";

                var textLoaderArgs = RecipeInference.MyAutoMlInferTextLoaderArguments(env,
                        trainDataPath, "Label");

                var trainData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, trainDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;
                var validData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, validDataPath, false, false),
                    Arguments = textLoaderArgs
                }).Data;
                var testData = ImportTextData.TextLoader(env, new ImportTextData.LoaderInput()
                {
                    InputFile = new SimpleFileHandle(env, testDataPath, false, false),
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

                File.AppendAllText($"{MyGlobals.OutputDir}/test_metric.txt", $"{testMetricVal}\r\n");
            }

            File.AppendAllText($"{MyGlobals.OutputDir}/time.txt", $"{stopwatch.ElapsedMilliseconds}ms\r\n");
        }
    }
}
