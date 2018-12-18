using System.Collections.Generic;
using Microsoft.ML.Runtime.PipelineInference;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.EntryPoints;
using static Microsoft.ML.Runtime.PipelineInference.AutoInference;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class AutoMlExperiment
    {
        private AutoMlConfig _config;
        private List<ITrainingIterationNotifications> _iterationOvservers;

        public AutoMlExperiment(AutoMlConfig config)
        {
            _config = config;
            _iterationOvservers = new List<ITrainingIterationNotifications>();
        }

       public AutoMlResult Fit(IHostEnvironment env, EstimatorChain<ITransformer> preprocessor = null)
        {
            // hack: init legacy assembly, for catalog of learners
            LegacyAssemblyUtil.Init();

            var rocketEngine = new RocketEngine(env, new RocketEngine.Arguments() { });

            var terminator = new IterationTerminator(_config.MaxIterationCount);

            var amls = new AutoMlMlState(
                env,
                PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.RSquared),
                rocketEngine,
                terminator,
                MacroUtils.TrainerKinds.SignatureRegressorTrainer,
                _config.TrainingData,
                _config.ValidationData);

            var resultsRecorder = new IterationResultRecorder();
            amls.IterationMonitor.Subscribe(resultsRecorder);
            _iterationOvservers.ForEach(o => amls.IterationMonitor.Subscribe(o));

            var bestPipelines = amls.InferPipelines(1, 1, 100);
            var bestPipeline = bestPipelines.First();

            // hack: start dummy host & channel
            var host = env.Register("hi");
            var ch = host.Start("hi");

            var transformer = bestPipeline.TrainTransformer(_config.TrainingData, ch);

            var result = new AutoMlResult(resultsRecorder.IterationResults, transformer);

            return result;
        }

        public void AddIterationObserver(ITrainingIterationNotifications observer)
        {
            _iterationOvservers.Add(observer);
        }
    }

    public class AutoMlConfig
    {
        public Dictionary<string, ColumnPurpose> ColumnPurposes;
        public int MaxIterationCount;
        public IDataView TrainingData;
        public IDataView ValidationData;
    }
}
