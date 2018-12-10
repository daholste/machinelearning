using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoMLPublicAPI
{
    public class LegacyAssemblyUtil
    {
        public static ConsoleEnvironment Env;

        private static IDisposable _registrar;

        public static void Init()
        {
            var dir = AppDomain.CurrentDomain.BaseDirectory;
            Env = new ConsoleEnvironment();
            _registrar = AssemblyLoadingUtils.CreateAssemblyRegistrar(Env, dir);
        }
    }
}
