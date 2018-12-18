using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using static Microsoft.ML.Runtime.PipelineInference.AutoInference;

namespace Microsoft.ML.PipelineInference
{
    public static class MyGlobals
    {
        public static string OutputDir = ".";
        public static ISet<string> FailedPipelineHashes = new HashSet<string>();
        public static Stopwatch Stopwatch = Stopwatch.StartNew();
    }
}
