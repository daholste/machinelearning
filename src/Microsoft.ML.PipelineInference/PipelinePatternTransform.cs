using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference
{
    public class PipelinePatternTransform : ITransformer
    {
        private readonly ITransformer _preprocessor;
        private readonly IPredictor _predictor;
        private readonly IHostEnvironment _env;

        private RowToRowScorerBase _scorer;

        internal PipelinePatternTransform(IHostEnvironment env, ITransformer preprocessor,
            IPredictor predictor)
        {
            _env = env;
            _preprocessor = preprocessor;
            _predictor = predictor;
        }

        public bool IsRowToRowMapper => true;

        public Schema GetOutputSchema(Schema inputSchema)
        {
            return Transform(new EmptyDataView(_env, inputSchema)).Schema;
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            var preprocessorMapper = _preprocessor.GetRowToRowMapper(inputSchema);
            return new CompositeRowToRowMapper(inputSchema,
                new[] { preprocessorMapper, _scorer });
        }

        public IDataView Transform(IDataView data)
        {
            data = _preprocessor.Transform(data);
            var roleMappedData = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                RoleMappedSchema.ColumnRole.Feature.Bind(DefaultColumnNames.Features));
            // add normalizers
            //TrainUtils.AddNormalizerIfNeeded(env, ch, learner, ref trainData, "Features", Data.NormalizeOption.Auto);
            //roleMappedTestData = ApplyTransformUtils.ApplyAllTransformsToData(env, scoredTestData, scoredTestData);
            _scorer = ScoreUtils.GetScorer(_predictor, roleMappedData, _env, roleMappedData.Schema) as RowToRowScorerBase;
            return _scorer.ApplyToData(_env, data);
        }
    }
}
