using Microsoft.ML.Core.Data;
using Microsoft.ML.PipelineInference;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Text;

[assembly: LoadableClass(typeof(PipelinePatternTransform), typeof(PipelinePatternTransform), null,
        typeof(SignatureLoadModel), "pipeline pattern transform", PipelinePatternTransform.LoaderSignature)]

namespace Microsoft.ML.PipelineInference
{
    public class PipelinePatternTransform : ITransformer, ICanSaveModel
    {
        public const string LoaderSignature = "PipelinePatternTransform";

        private readonly ITransformer _preprocessor;
        private readonly IPredictor _predictor;
        private readonly IHostEnvironment _env;

        internal PipelinePatternTransform(IHostEnvironment env, ITransformer preprocessor,
            IPredictor predictor)
        {
            _env = env;
            _preprocessor = preprocessor;
            _predictor = predictor;
        }

        internal PipelinePatternTransform(IHostEnvironment env, ModelLoadContext ctx)
        {
            _env = env;
            ctx.LoadModel<ITransformer, SignatureLoadModel>(env, out _preprocessor, "Preprocessor");
            ctx.LoadModel<IPredictor, SignatureLoadModel>(env, out _predictor, "Predictor");
        }

        public bool IsRowToRowMapper => true;

        public Schema GetOutputSchema(Schema inputSchema)
        {
            return Transform(new EmptyDataView(_env, inputSchema)).Schema;
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            IDataView data = new EmptyDataView(_env, inputSchema);
            var scorer = BuildScorer(ref data);

            var preprocessorMapper = _preprocessor.GetRowToRowMapper(inputSchema);
            return new CompositeRowToRowMapper(inputSchema,
                new[] { preprocessorMapper, scorer });
        }

        public IDataView Transform(IDataView data)
        {
            var scorer = BuildScorer(ref data);
            return scorer.ApplyToData(_env, data);
        }

        private RowToRowScorerBase BuildScorer(ref IDataView data)
        {
            data = _preprocessor.Transform(data);
            var roleMappedData = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                RoleMappedSchema.ColumnRole.Feature.Bind(DefaultColumnNames.Features));
            // add normalizers
            //TrainUtils.AddNormalizerIfNeeded(env, ch, learner, ref trainData, "Features", Data.NormalizeOption.Auto);
            //roleMappedTestData = ApplyTransformUtils.ApplyAllTransformsToData(env, scoredTestData, scoredTestData);
            var scorer = ScoreUtils.GetScorer(_predictor, roleMappedData, _env, roleMappedData.Schema) as RowToRowScorerBase;
            return scorer;
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            ctx.SaveModel(_preprocessor, "Preprocessor");
            ctx.SaveModel(_predictor, "Predictor");
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PIPELINE",
                verWrittenCur: 0,
                verReadableCur: 0,
                verWeCanReadBack: 0,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PipelinePatternTransform).Assembly.FullName);
        }
    }
}
