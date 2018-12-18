// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public interface ITrainingIterationNotifications
    {
        void IterationStarted(Iteration iteration);

        void IterationFinished(IterationResult result);

        void IterationFailed(Exception e);
    }

    public class Iteration
    {
        public PipelinePattern Pipeline;
    }

    public class IterationResult
    {
        public PipelinePattern Pipeline;

        public RegressionEvaluator.Result Score;

        public long  TimeElapsedInMilliseconds;
    }

    public class IterationMonitor : ITrainingIterationNotifications
    {
        private readonly List<ITrainingIterationNotifications> _observers;

        public IterationMonitor()
        {
             _observers = new List<ITrainingIterationNotifications>();
        }

        public void Subscribe(ITrainingIterationNotifications observer)
        {
            _observers.Add(observer);
        }

        public void IterationFinished(IterationResult result)
        {
            foreach(var nextObserver in _observers)
            {
                nextObserver.IterationFinished(result);
            }
        }

        public void IterationStarted(Iteration iteration)
        {
            foreach (var nextObserver in _observers)
            {
                nextObserver.IterationStarted(iteration);
            }
        }

        public void IterationFailed(Exception e)
        {
            foreach (var nextObserver in _observers)
            {
                nextObserver.IterationFailed(e);
            }
        }
    }

    public class IterationResultRecorder : ITrainingIterationNotifications
    {
        public List<IterationResult> IterationResults;

        public IterationResultRecorder()
        {
            IterationResults = new List<IterationResult>();
        }

        public void IterationFinished(IterationResult result)
        {
            IterationResults.Add(result);
        }

        public void IterationStarted(Iteration iteration)
        {
        }
        public void IterationFailed(Exception e)
        {
        }
    }
}