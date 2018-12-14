
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.PipelineInference2
{
    public sealed class SimpleComponentFactory<TComponent> : IComponentFactory<TComponent>
    {
        private readonly Func<IHostEnvironment, TComponent> _factory;

        public SimpleComponentFactory(Func<IHostEnvironment, TComponent> factory)
        {
            _factory = factory;
        }

        public TComponent CreateComponent(IHostEnvironment env)
        {
            return _factory(env);
        }
    }

    /// <summary>
    /// A utility class for creating <see cref="IComponentFactory"/> instances.
    /// </summary>
    public static class ComponentFactoryUtils
    {
        /// <summary>
        /// Creates a component factory with no extra parameters (other than an <see cref="IHostEnvironment"/>)
        /// that simply wraps a delegate which creates the component.
        /// </summary>
        public static IComponentFactory<TComponent> CreateFromFunction<TComponent>(Func<IHostEnvironment, TComponent> factory)
        {
            return new SimpleComponentFactory<TComponent>(factory);
        }
    }
}
