[StochasticTools]
[]

[Distributions]
  [k_dist]
    type = Uniform
    lower_bound = 1
    upper_bound = 10
  []
  [q_dist]
    type = Uniform
    lower_bound = 9000
    upper_bound = 11000
  []
[]

[Samplers]
  [train_sample]
    type = MonteCarlo
    num_rows = 20
    distributions = 'k_dist q_dist'
    execute_on = PRE_MULTIAPP_SETUP
    seed = 100
  []
  [test_sample]
    type = MonteCarlo
    num_rows = 100
    distributions = 'k_dist q_dist'
    execute_on = PRE_MULTIAPP_SETUP
    seed = 100
  []
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    input_files = sub.i
    sampler = train_sample
  []
[]

[Controls]
  [cmdline]
    type = MultiAppSamplerControl
    multi_app = sub
    sampler = train_sample
    param_names = 'Materials/conductivity/prop_values Kernels/source/value'
  []
[]

[Transfers]
  [data]
    type = SamplerReporterTransfer
    from_multi_app = sub
    sampler = train_sample
    stochastic_reporter = results
    from_reporter = 'avg/value'
  []
[]

[Reporters]
  [results]
    type = StochasticReporter
    parallel_type = ROOT
  []
  [samp_avg]
    type = EvaluateSurrogate
    model = DGP_avg
    sampler = test_sample
    evaluate_std = 'true'
    parallel_type = ROOT
    execute_on = final
  []
  [train_avg]
    type = EvaluateSurrogate
    model = DGP_avg
    sampler = train_sample
    evaluate_std = 'true'
    parallel_type = ROOT
    execute_on = final
  []
[]

[VectorPostprocessors]
  [hyperparams]
    type = GaussianProcessData
    gp_name = 'DGP_avg'
    execute_on = final
  []
  [data]
    type = SamplerData
    sampler = test_sample
    execute_on = 'initial timestep_end'
  []
[]

[Trainers]
  [DGP_avg_trainer]
    type = DeepGaussianProcessTrainer
    execute_on = timestep_end
    covariance_function = 'covar' #Choose a squared exponential for the kernel
    standardize_params = 'true' #Center and scale the training params
    standardize_data = 'true' #Center and scale the training data
    sampler = train_sample
    response = results/data:avg:value
    tune_parameters = 'covar:signal_variance covar:length_factor'
    num_iters = 1000
    batch_size = 20
    learning_rate = 0.005
    show_optimization_details = false
  []
[]

[Surrogates]
  [DGP_avg]
    type = DeepGaussianProcessSurrogate
    trainer = DGP_avg_trainer
  []
[]

[Covariance]
  [covar]
    type = SquaredExponentialCovariance
    signal_variance = 1.0 #Use a signal variance of 1 in the kernel
    noise_variance = 1e-6 #A small amount of noise can help with numerical stability
    length_factor = '1.0 1.0' #Select a length factor for each parameter (k and q)
  []
[]

[Outputs]
  [out]
    type = CSV
    execute_on = FINAL
  []
[]
