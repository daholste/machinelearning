using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.PipelineInference
{
    public static class MyGlobals
    {
        public static string DatasetName;
        public static ISet<string> FailedPipelineHashes = new HashSet<string>();
        public static Stopwatch Stopwatch;
    }
}
