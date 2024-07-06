//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "DeepGaussianProcess.h"
#include "FEProblemBase.h"

#include <petsctao.h>
#include <petscdmda.h>

#include "libmesh/petsc_vector.h"
#include "libmesh/petsc_matrix.h"

#include <cmath>

#include "MooseRandom.h"
#include "Shuffle.h"

// #include <random>
#include "Uniform.h"
#include "Gamma.h"

namespace StochasticTools
{

DeepGaussianProcess::DGPOptimizerOptions::DGPOptimizerOptions(const bool inp_show_optimization_details,
                                                        const unsigned int inp_num_iter,
                                                        const unsigned int inp_batch_size,
                                                        const Real inp_learning_rate,
                                                        const Real inp_b1,
                                                        const Real inp_b2,
                                                        const Real inp_eps,
                                                        const Real inp_lambda)
  : show_optimization_details(inp_show_optimization_details),
    num_iter(inp_num_iter),
    batch_size(inp_batch_size),
    learning_rate(inp_learning_rate),
    b1(inp_b1),
    b2(inp_b2),
    eps(inp_eps),
    lambda(inp_lambda)
{
}

DeepGaussianProcess::DeepGaussianProcess() {}

void
DeepGaussianProcess::initialize(CovarianceFunctionBase * covariance_function,
                            const std::vector<std::string> & params_to_tune,
                            const std::vector<Real> & min,
                            const std::vector<Real> & max)
{
  linkCovarianceFunction(covariance_function);
  generateTuningMap(params_to_tune, min, max);
}

void
DeepGaussianProcess::linkCovarianceFunction(CovarianceFunctionBase * covariance_function)
{
  _covariance_function = covariance_function;
  _covar_type = _covariance_function->type();
  _covar_name = _covariance_function->name();
  _covariance_function->dependentCovarianceTypes(_dependent_covar_types);
  _dependent_covar_names = _covariance_function->dependentCovarianceNames();
  _num_outputs = _covariance_function->numOutputs();
}

void
DeepGaussianProcess::setupCovarianceMatrix(const RealEigenMatrix & training_params,
                                       const RealEigenMatrix & training_data,
                                       const DGPOptimizerOptions & opts)
{
  const bool batch_decision = opts.batch_size > 0 && (opts.batch_size <= training_params.rows());
  _batch_size = batch_decision ? opts.batch_size : training_params.rows();
  _K.resize(_num_outputs * _batch_size, _num_outputs * _batch_size);

  std::cout << "_K shape is" << _K.rows() << "," << _K.cols() << "," << _num_outputs << "," << _batch_size << std::endl;

  if (_tuning_data.size())
    // tuneHyperParamsAdam(training_params, training_data, opts);
    tuneHyperParamsMcmc(training_params, training_data, opts);


  _K.resize(training_params.rows() * training_data.cols(),
            training_params.rows() * training_data.cols());
  _covariance_function->computeCovarianceMatrix(_K, training_params, training_params, true);

  RealEigenMatrix flattened_data =
      training_data.reshaped(training_params.rows() * training_data.cols(), 1);

  // Compute the Cholesky decomposition and inverse action of the covariance matrix
  setupStoredMatrices(flattened_data);

  _covariance_function->buildHyperParamMap(_hyperparam_map, _hyperparam_vec_map);
}

void
DeepGaussianProcess::setupStoredMatrices(const RealEigenMatrix & input)
{
  _K_cho_decomp = _K.llt();
  _K_results_solve = _K_cho_decomp.solve(input);
}

void
DeepGaussianProcess::generateTuningMap(const std::vector<std::string> & params_to_tune,
                                   const std::vector<Real> & min_vector,
                                   const std::vector<Real> & max_vector)
{
  _num_tunable = 0;

  const bool upper_bounds_specified = min_vector.size();
  const bool lower_bounds_specified = max_vector.size();

  for (const auto param_i : index_range(params_to_tune))
  {
    const auto & hp = params_to_tune[param_i];
    if (_covariance_function->isTunable(hp))
    {
      unsigned int size;
      Real min;
      Real max;
      // Get size and default min/max
      const bool found = _covariance_function->getTuningData(hp, size, min, max);

      if (!found)
        ::mooseError("The covariance parameter ", hp, " could not be found!");

      // Check for overridden min/max
      min = lower_bounds_specified ? min_vector[param_i] : min;
      max = upper_bounds_specified ? max_vector[param_i] : max;
      // Save data in tuple
      _tuning_data[hp] = std::make_tuple(_num_tunable, size, min, max);
      _num_tunable += size;
    }
  }
}

void
DeepGaussianProcess::standardizeParameters(RealEigenMatrix & data, bool keep_moments)
{
  if (!keep_moments)
    _param_standardizer.computeSet(data);
  _param_standardizer.getStandardized(data);
}

void
DeepGaussianProcess::standardizeData(RealEigenMatrix & data, bool keep_moments)
{
  if (!keep_moments)
    _data_standardizer.computeSet(data);
  _data_standardizer.getStandardized(data);
}

void 
DeepGaussianProcess::sq_dist(const RealEigenMatrix &X1_in, RealEigenMatrix &D_out, const RealEigenMatrix &X2_in) {
  if (X2_in.size() == 0) {
    std::cout << "enter sym" << std::endl;
    int n = X1_in.rows();
    int m = X1_in.cols();

    D_out.resize(n, n);
    D_out.setZero();

    for (int i = 0; i < n; ++i) {
      D_out(i, i) = 0.0;
      for (int j = i + 1; j < n; ++j) {
          D_out(i, j) = (X1_in.row(i) - X1_in.row(j)).squaredNorm();
          D_out(j, i) = D_out(i, j);
      }
    }
  } else {
    std::cout << "enter nosym" << std::endl;
    int n1 = X1_in.rows();
    int m = X1_in.cols();
    int n2 = X2_in.rows();

    D_out.resize(n1, n2);
    D_out.setZero();

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            D_out(i, j) = (X1_in.row(i) - X2_in.row(j)).squaredNorm();
        }
    }
  }
}

void 
DeepGaussianProcess::Exp2(const RealEigenMatrix & distmat, Real tau2, Real theta, Real g, RealEigenMatrix & covmat) {
  int n1 = distmat.rows();
  int n2 = distmat.cols();
  covmat.resize(n1, n2);
  
  for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n2; ++j) {
          Real r = distmat(i, j) / theta;
          covmat(i, j) = tau2 * std::exp(-r);
      }
  }

  if (n1 == n2) {
      for (int i = 0; i < n1; ++i) {
          covmat(i, i) += tau2 * g;
      }
  }
}

void 
DeepGaussianProcess::squared_exponential_covariance(const RealEigenMatrix &x1, 
                  const RealEigenMatrix &x2, 
                  Real tau2, 
                  const RealEigenMatrix &theta, 
                  Real g, 
                  RealEigenMatrix &k){
  int n1 = x1.rows();
  int n2 = x2.rows();
  // k.resize(n1, n2);
  // std::cout << "x1 rows is " << x1.rows() << std::endl;
  // std::cout << "x1 cols is " << x1.cols() << std::endl;
  
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      // Compute the scaled distance r_l(x1, x2)
      Eigen::RowVectorXd diff = (x1.row(i) - x2.row(j)).array() / theta.row(0).array();
      Real r_l = std::sqrt(diff.squaredNorm());
      Real cov_val = tau2 * std::exp(-0.5 * r_l * r_l);
      if (i == j) {
          cov_val += g;
      }
      k(i, j) = cov_val;
    }
  }
  // std::cout << "k is " << k << std::endl;
}
                                    

void 
DeepGaussianProcess::inv_det(const RealEigenMatrix & M, InvDetResult & result) {
  Eigen::LLT<RealEigenMatrix> llt(M);
  if (llt.info() == Eigen::NumericalIssue) {
      throw std::runtime_error("Matrix is not positive definite");
  }
  RealEigenMatrix Mi = llt.solve(RealEigenMatrix::Identity(M.rows(), M.cols()));
  RealEigenMatrix L = llt.matrixL();
  Real ldet = 2 * L.diagonal().array().log().sum();
  result.Mi = Mi;
  result.ldet = ldet;
}

void
DeepGaussianProcess::logl(const RealEigenMatrix & out_vec, const RealEigenMatrix & in_dmat, const RealEigenMatrix & x1, const RealEigenMatrix & x2, Real g, const RealEigenMatrix & theta, 
          LogLResult & result, bool outer, bool tau2, Real mu, Real scale) {
  std::cout << "enter logl" << std::endl;
  int n = out_vec.rows();
  RealEigenMatrix K(x1.rows(), x2.rows());
  squared_exponential_covariance(x1, x2, 1, theta, g, K);

  // _covariance_function->computeCovarianceMatrix(_K, x1, x2, true);

  // K = scale * K;

  // std::cout << "K is " << K << std::endl;

  // _K_cho_decomp = _K.llt();
  // _K_results_solve = _K_cho_decomp.solve(input);


  InvDetResult inv_det_result;
  inv_det(K, inv_det_result);
  RealEigenMatrix Mi = inv_det_result.Mi;
  Real ldet = inv_det_result.ldet;

  // std::cout << "Mi is " << Mi << std::endl;
  // std::cout << "Mi shape is " << Mi.rows() << " x " << Mi.cols() << std::endl;

  // std::cout << "ldet is " << ldet << std::endl;


  RealEigenMatrix diff = out_vec;
  // std::cout << "diff is" << diff << std::endl;
  // std::cout << "diff shape is" << out_vec.rows() << " x " << out_vec.cols() << std::endl;
  // std::cout << "quadterm shape is" << (diff.transpose() * Mi * diff).rows() << " x " <<  (diff.transpose() * Mi * diff).cols() << std::endl;
  Real quadterm = (diff.transpose() * Mi * diff)(0,0);
  // std::cout << "quadterm is " << quadterm << std::endl;


  Real logl_val;
  if (outer) {
      logl_val = (-n * 0.5) * std::log(quadterm) - 0.5 * ldet;
  } else {
      logl_val = -0.5 * quadterm - 0.5 * ldet;
  }
  // std::cout << "logl_val is " << logl_val << std::endl;


  Real tau2_val;
  if (tau2) {
      tau2_val = quadterm / n;
  } else {
      tau2_val = NAN;
  }

  // std::cout << "tau2_val is " << tau2_val << std::endl;

  result.logl = logl_val;
  result.tau2 = tau2_val; 
}

void
DeepGaussianProcess::sample_g(const RealEigenMatrix & out_vec, const RealEigenMatrix & in_dmat, const RealEigenMatrix & x1, const RealEigenMatrix & x2, Real g_t, const RealEigenMatrix theta, 
              Real alpha, Real beta, Real l, Real u, Real ll_prev, SampleGResult & result, unsigned int j) {

  // propose
  // std::uniform_real_distribution<> runif(0, 1);
  // std::uniform_real_distribution<> runif_g(l * g_t / u, u * g_t / l);
  
  // MooseRandom generator1;
  // generator1.seed(0, 1980);
  // generator1.saveState();

  // Real unif_rnd = MooseRandom::rand();
  // Real gamma_rnd = Gamma::quantile(unif_rnd, alpha, beta);

  // calculate threshold

  Real ru1 = MooseRandom::rand();
  
  Real ru = MooseRandom::rand();

  Real g_star = Uniform::quantile(ru1, l * g_t / u, u * g_t / l);

  // Real ru = 0.7;
  // Real g_star = 0.02;

  // if (j==1){
  //   ru = 0.7203244934421581;
  //   g_star = 0.01125533007053861;
  // }
  // else if (j==2){
  //   ru = 0.0923385947687978;
  //   g_star = 0.008105344021683105;
  // }
  // else if (j==3){
  //   g_star = 0.008876577323722351;
  //   ru = 0.538816734003357;
  // }
  
  std::cout << "ru is " << ru << std::endl;
  std::cout << "g_star is " << g_star << std::endl;


  if (std::isnan(ll_prev)) {
    std::cout << "enter ll_prev nan" << ll_prev << std::endl;
    LogLResult ll_result;
    logl(out_vec, in_dmat, x1, x2, g_t, theta, ll_result, true, false);
    ll_prev = ll_result.logl;
    std::cout << "ll_prev is " << ll_prev << std::endl;
  }
  Real lpost_threshold = ll_prev + std::log(Gamma::pdf(g_t - 1.5e-8, alpha, beta)) + 
                            std::log(ru) - std::log(g_t) + std::log(g_star);

  std::cout << "lpost_threshold is " << lpost_threshold << std::endl;


  Real ll_new;
  LogLResult ll_result;
  logl(out_vec, in_dmat, x1, x2, g_star, theta, ll_result, true, false);
  ll_new = ll_result.logl;

  std::cout << "ll_new is " << ll_new << std::endl;

  // accept or reject
  Real new_val = ll_new + std::log(Gamma::pdf(g_star - 1.5e-8, alpha, beta));

  std::cout << "new_val is " << new_val << std::endl;

  
  if (new_val > lpost_threshold) {
    result.g = g_star;
    result.ll = ll_new;
  } else {
    result.g = g_t;
    result.ll = ll_prev;
  }
}

void
DeepGaussianProcess::sample_theta(const RealEigenMatrix & out_vec, const RealEigenMatrix & in_dmat, const RealEigenMatrix & x1, const RealEigenMatrix & x2, Real g, const RealEigenMatrix & theta_t,
              unsigned int i, Real alpha, Real beta, Real l, Real u, bool outer, SampleThetaResult & result, unsigned int j, Real ll_prev, bool tau2, 
              Real prior_mean, Real scale) {

    // MooseRandom generator2;
    // generator2.seed(0, 1980);
    // generator2.saveState();
    RealEigenMatrix theta_star = theta_t;
    // Propose value
    // Compute acceptance threshold
    Real ru1 = MooseRandom::rand();
    Real ru = MooseRandom::rand();
    theta_star(0, i) = Uniform::quantile(ru1, l * theta_t(0,i) / u, u * theta_t(0,i) / l);


    // Real ru = 0.7;
    // Real theta_star = 0.6;

    // if (j==1){
    //   theta_star = 0.25008578111300866;
    //   ru = 0.30233257263183977;
    // }
    // else if (j==2){
    //   theta_star = 0.3896951585332532;
    //   ru = 0.34556072704304774;
    // }
    // else if (j==3){
    //   theta_star = 0.43988468838661965;
    //   ru = 0.6852195003967595;
    // }


    if (std::isnan(ll_prev)) {
      LogLResult ll_result;
      logl(out_vec, in_dmat, x1, x2, g, theta_t, ll_result, true, true);
      ll_prev = ll_result.logl;
      std::cout << "theta ll_prev is " << ll_prev << std::endl;
    }
              
    Real lpost_threshold = ll_prev + std::log(Gamma::pdf(theta_t(0,i), alpha, beta)) + 
                             std::log(ru) - std::log(theta_t(0,i)) + std::log(theta_star(0,i));
    std::cout << "theta lpost_threshold is " << lpost_threshold << std::endl;

    
    Real ll_new;
    Real tau2_new;
    LogLResult ll_result;
    logl(out_vec, in_dmat, x1, x2, g, theta_star, ll_result, true, true);
    ll_new = ll_result.logl;
    tau2_new = ll_result.tau2;

    std::cout << "theta ll_new tau2_new is " << ll_new << " " << tau2_new << std::endl;

      
    // Accept or reject (lower bound of eps)
    Real new_val = ll_new + std::log(Gamma::pdf(theta_star(0,i), alpha, beta));
    if (new_val > lpost_threshold) {
      result.theta = theta_star(0,i);
      result.ll = ll_new;
      result.tau2 = tau2_new;
    } else {
      result.theta = theta_t(0,i);
      result.ll = ll_prev;
      result.tau2 = NAN;
    }
}


void 
DeepGaussianProcess::check_settings(Settings &settings) {
  settings.l = 1;
  settings.u = 2;

  settings.alpha.g = 1.5;
  settings.beta.g = 3.9;

  settings.alpha.theta = 1.5;
  settings.beta.theta = 3.9 / 1.5;
}


void
DeepGaussianProcess::tuneHyperParamsMcmc(const RealEigenMatrix & training_params,
                                     const RealEigenMatrix & training_data,
                                     const DGPOptimizerOptions & opts)
{ 
  std::cout << "enter dgp mcmc " << std::endl;

  std::cout << "training params is" << training_params << std::endl;
  std::cout << "training data is" << training_data << std::endl;

  for (const auto & pair : _tuning_data){
    std::cout << pair.first << "-- " << pair.second << std::endl;
  }

  // std::unordered_map<std::string, Real> _hyperparam_map;
  // std::unordered_map<std::string, std::vector<Real>> _hyperparam_vec_map;

  for (const auto & pair : _hyperparam_map){
    std::cout << pair.first << ":::hhh0 " << pair.second << std::endl;
  }

  for (const auto & pair : _hyperparam_vec_map){
    std::cout << pair.first << ":::hhh0 " << Moose::stringify(pair.second) << std::endl;
  }

  std::vector<Real> theta1(_num_tunable, 0.0);
  _covariance_function->buildHyperParamMap(_hyperparam_map, _hyperparam_vec_map);

  for (const auto & pair : _hyperparam_map){
    std::cout << pair.first << ":::hhh " << pair.second << std::endl;
  }

  for (const auto & pair : _hyperparam_vec_map){
    std::cout << pair.first << ":::hhh " << Moose::stringify(pair.second) << std::endl;
  }

  mapToVec(_tuning_data, _hyperparam_map, _hyperparam_vec_map, theta1);

  std:: cout << "map to vector";
  for (const auto& value : theta1){
    std::cout << value << " ";
  }
  std::cout << std::endl;

  RealEigenMatrix x = training_params;
  RealEigenMatrix y = training_data;

  // std::cout << "print hyper0" << std::endl;
  // _covariance_function->computeCovarianceMatrix(_K, x, x, true);
  // std::cout << "_length_factor _sigma_f_squared _sigma_n_squared 0 is" << _length_factor << "," << _sigma_f_squared << std::endl;

  
  MooseRandom generator;
  generator.seed(0, 1980);
  Real h1 = MooseRandom::rand();
  std::cout << "h1 is " << h1 << std::endl;
  // std::default_random_engine generator;
  // generator.seed(2);
  unsigned int layers = 1;
  unsigned int n = training_params.rows();
  std::cout << "training_params rows" << ": " << n << std::endl;
  unsigned int new_n = 40;
  unsigned int m = 100;
  Real noise = 0.1;

  // const RealEigenMatrix & training_params;
  // const RealEigenMatrix & training_data;

  
  unsigned int nmcmc = 10000;
  unsigned int burn = 8000;
  unsigned int thin = 2;
  unsigned int D = training_params.cols();
  std::cout << "training_params cols" << ": " << D << std::endl;

  Settings settings;
  check_settings(settings);

  std::cout << "l: " << settings.l << std::endl;
  std::cout << "u: " << settings.u << std::endl;
  std::cout << "alpha.g: " << settings.alpha.g << std::endl;
  std::cout << "beta.g: " << settings.beta.g << std::endl;
  std::cout << "alpha.theta: " << settings.alpha.theta << std::endl;
  std::cout << "beta.theta: " << settings.beta.theta << std::endl;

  // RealEigenMatrix x = training_params;
  // RealEigenMatrix y = training_data;
  // RealEigenMatrix x(2,2);
  // x << 1,2,
  //      3,4;
  // RealEigenMatrix y(2,1);
  // y << 3,
  //      7;
  std::cout << "matrix x:\n" << x << std::endl;
  std::cout << "matrix y:\n" << y << std::endl;
  // sq_dist(X1_in, D_out);
  // std::cout << "distance matrix" << ": " << D_out << std::endl;
  // sq_dist(X1_in, D_out, X2_in);
  // std::cout << "distance matrix" << ": " << D_out << std::endl;


  // Set initial values for MCMC
  Real g_0 = 0.01;
  // Real theta_0 = 0.5;
  RealEigenMatrix theta_0(1,x.cols());
  theta_0 << 0.5,0.5;
  Real tau_0 = 1;

  Initial initial = {theta_0, g_0, tau_0};
  std::cout << "theta: " << initial.theta << std::endl;
  std::cout << "g: " << initial.g << std::endl;
  std::cout << "tau2: " << initial.tau2 << std::endl;


  Output out;
  out.x = x;
  out.y = y;
  out.nmcmc = nmcmc;
  out.initial = initial;
  out.settings = settings;
  RealEigenMatrix dx;
  sq_dist(x, dx);
  RealEigenMatrix g(nmcmc, 1);
  g(0,0) = initial.g;
  RealEigenMatrix theta(nmcmc, x.cols());
  theta.row(0) = initial.theta;
  RealEigenMatrix tau2(nmcmc, 1);
  tau2(0,0) = initial.tau2;
  RealEigenMatrix ll_store(nmcmc, x.cols());
  ll_store.row(0) << NAN, NAN;
  Real ll = NAN;
  
  for (unsigned int j = 1; j < nmcmc; ++j) {
    if (j % 500 == 0) {
        std::cout << "round" << j << std::endl;
    }

    // Sample nugget (g)

    SampleGResult sample_g_result;
    sample_g(y, dx, x, x, g(j-1,0), theta.row(j-1), settings.alpha.g, settings.beta.g, settings.l, settings.u, ll, sample_g_result, j);

    g(j,0) = sample_g_result.g;
    ll = sample_g_result.ll;

    for (unsigned int i=0; i<x.cols(); i++){
      SampleThetaResult sample_theta_result;
      sample_theta(y, dx, x, x, g(j,0), theta.row(j-1), i, settings.alpha.theta, settings.beta.theta, settings.l,
                  settings.u, true, sample_theta_result, j, ll, true);
      theta(j,i) = sample_theta_result.theta;
      ll = sample_theta_result.ll;
      ll_store(j,i) = ll;
      if (std::isnan(sample_theta_result.tau2)) {
        tau2(j,0) = tau2(j-1,0);
      }
      else {
        tau2(j,0) = sample_theta_result.tau2;
      }
    }
    

    std::cout << "g is: " << g(j,0) << std::endl;
    std::cout << "theta is: " << theta.row(j) << std::endl;
    std::cout << "tau2 is: " << tau2(j,0) << std::endl;
    std::cout << "ll is: " << ll_store.row(j) << std::endl;
    std::cout << std::endl;

    // theta1[0] = tau2(j,0);
    // theta1[1] = theta(j,0);
    // theta1[2] = theta(j,1);

    // theta1[0] = 1.117458397680367;
    // theta1[1] = 0.50030703;
    // theta1[2] = 5.8918648;

    theta1[0] = 1;
    theta1[1] = 1;
    theta1[2] = 1;


    // std:: cout << "theta global:" << std::endl;
    // for (const auto& value : theta1){
    //   std::cout << value << " ";
    // }
    // std::cout << "theta global end" << std::endl;

    vecToMap(_tuning_data, _hyperparam_map, _hyperparam_vec_map, theta1);
    _covariance_function->loadHyperParamMap(_hyperparam_map, _hyperparam_vec_map);

    // const std::vector<Real> & _length_factor;
    // const Real & _sigma_f_squared;
    // const Real & _sigma_n_squared;
    // std::cout << "print hyper1" << std::endl;
    // _covariance_function->computeCovarianceMatrix(_K, x, x, true);
    // std::cout << "_length_factor _sigma_f_squared _sigma_n_squared 1 is" << _length_factor << "," << _sigma_f_squared << std::endl;

  

  }
}

void
DeepGaussianProcess::tuneHyperParamsAdam(const RealEigenMatrix & training_params,
                                     const RealEigenMatrix & training_data,
                                     const DGPOptimizerOptions & opts)
{
  std::cout << "training params is" << training_params << std::endl;
  std::cout << "training data is" << training_data << std::endl;
  std::cout << "gp option is" << opts.num_iter << "," << opts.batch_size << "," <<opts.learning_rate << std::endl;
  std::cout << "enter adam " << std::endl;
  for (const auto & pair : _tuning_data){
    std::cout << pair.first << "-- " << pair.second << std::endl;
  }

  std::vector<Real> theta(_num_tunable, 0.0);
  _covariance_function->buildHyperParamMap(_hyperparam_map, _hyperparam_vec_map);

  mapToVec(_tuning_data, _hyperparam_map, _hyperparam_vec_map, theta);

  std:: cout << "map to vector";
  for (const auto& value : theta){
    std::cout << value << " ";
  }
  std::cout << std::endl;


  // Internal params for Adam; set to the recommended values in the paper
  Real b1 = opts.b1;
  Real b2 = opts.b2;
  Real eps = opts.eps;

  std::vector<Real> m0(_num_tunable, 0.0);
  std::vector<Real> v0(_num_tunable, 0.0);

  Real new_val;
  Real m_hat;
  Real v_hat;
  Real store_loss = 0.0;
  std::vector<Real> grad1;

  // Initialize randomizer
  std::vector<unsigned int> v_sequence(training_params.rows());
  std::iota(std::begin(v_sequence), std::end(v_sequence), 0);
  RealEigenMatrix inputs(_batch_size, training_params.cols());
  RealEigenMatrix outputs(_batch_size, training_data.cols());
  if (opts.show_optimization_details)
    Moose::out << "OPTIMIZING GP HYPER-PARAMETERS USING Adam" << std::endl;
  for (unsigned int ss = 0; ss < opts.num_iter; ++ss)
  {
    // Shuffle data
    MooseRandom generator;
    generator.seed(0, 1980);
    generator.saveState();
    MooseUtils::shuffle<unsigned int>(v_sequence, generator, 0);
    for (unsigned int ii = 0; ii < _batch_size; ++ii)
    {
      for (unsigned int jj = 0; jj < training_params.cols(); ++jj)
        inputs(ii, jj) = training_params(v_sequence[ii], jj);

      for (unsigned int jj = 0; jj < training_data.cols(); ++jj)
        outputs(ii, jj) = training_data(v_sequence[ii], jj);
    }

    store_loss = getLoss(inputs, outputs);
    if (opts.show_optimization_details)
      Moose::out << "Iteration: " << ss + 1 << " LOSS: " << store_loss << std::endl;
    grad1 = getGradient(inputs);
    for (auto iter = _tuning_data.begin(); iter != _tuning_data.end(); ++iter)
    {
      const auto first_index = std::get<0>(iter->second);
      const auto num_entries = std::get<1>(iter->second);
      for (unsigned int ii = 0; ii < num_entries; ++ii)
      {
        const auto global_index = first_index + ii;
        m0[global_index] = b1 * m0[global_index] + (1 - b1) * grad1[global_index];
        v0[global_index] =
            b2 * v0[global_index] + (1 - b2) * grad1[global_index] * grad1[global_index];
        m_hat = m0[global_index] / (1 - std::pow(b1, (ss + 1)));
        v_hat = v0[global_index] / (1 - std::pow(b2, (ss + 1)));
        new_val = theta[global_index] - opts.learning_rate * m_hat / (std::sqrt(v_hat) + eps);

        const auto min_value = std::get<2>(iter->second);
        const auto max_value = std::get<3>(iter->second);

        theta[global_index] = std::min(std::max(new_val, min_value), max_value);
      }
    }
    std:: cout << "theta global:" << std::endl;
    for (const auto& value : theta){
      std::cout << value << " ";
    }
    std::cout << "theta global end" << std::endl;

    vecToMap(_tuning_data, _hyperparam_map, _hyperparam_vec_map, theta);
    _covariance_function->loadHyperParamMap(_hyperparam_map, _hyperparam_vec_map);

    std:: cout << "_tuning_data global:" << std::endl;
    for (const auto & pair : _tuning_data){
      std::cout << pair.first << "-- " << pair.second << std::endl;
    std::cout << "_tuning_data global end" << std::endl;
    }

  }
  if (opts.show_optimization_details)
  {
    Moose::out << "OPTIMIZED GP HYPER-PARAMETERS:" << std::endl;
    Moose::out << Moose::stringify(theta) << std::endl;
    Moose::out << "FINAL LOSS: " << store_loss << std::endl;
  }
}

Real
DeepGaussianProcess::getLoss(RealEigenMatrix & inputs, RealEigenMatrix & outputs)
{
  _covariance_function->computeCovarianceMatrix(_K, inputs, inputs, true);

  RealEigenMatrix flattened_data = outputs.reshaped(outputs.rows() * outputs.cols(), 1);

  setupStoredMatrices(flattened_data);

  Real log_likelihood = 0;
  log_likelihood += -(flattened_data.transpose() * _K_results_solve)(0, 0);
  log_likelihood += -std::log(_K.determinant());
  log_likelihood -= _batch_size * std::log(2 * M_PI);
  log_likelihood = -log_likelihood / 2;
  return log_likelihood;
}

std::vector<Real>
DeepGaussianProcess::getGradient(RealEigenMatrix & inputs)
{
  RealEigenMatrix dKdhp(_batch_size, _batch_size);
  RealEigenMatrix alpha = _K_results_solve * _K_results_solve.transpose();
  std::vector<Real> grad_vec;
  grad_vec.resize(_num_tunable);
  for (auto iter = _tuning_data.begin(); iter != _tuning_data.end(); ++iter)
  {
    std::string hyper_param_name = iter->first;
    const auto first_index = std::get<0>(iter->second);
    const auto num_entries = std::get<1>(iter->second);
    for (unsigned int ii = 0; ii < num_entries; ++ii)
    {
      const auto global_index = first_index + ii;
      _covariance_function->computedKdhyper(dKdhp, inputs, hyper_param_name, ii);
      RealEigenMatrix tmp = alpha * dKdhp - _K_cho_decomp.solve(dKdhp);
      grad_vec[global_index] = -tmp.trace() / 2.0;
    }
  }
  return grad_vec;
}

void
DeepGaussianProcess::mapToVec(
    const std::unordered_map<std::string, std::tuple<unsigned int, unsigned int, Real, Real>> &
        tuning_data,
    const std::unordered_map<std::string, Real> & scalar_map,
    const std::unordered_map<std::string, std::vector<Real>> & vector_map,
    std::vector<Real> & vec)
{
  for (auto iter : tuning_data)
  {
    const std::string & param_name = iter.first;
    const auto scalar_it = scalar_map.find(param_name);
    if (scalar_it != scalar_map.end())
      vec[std::get<0>(iter.second)] = scalar_it->second;
    else
    {
      const auto vector_it = vector_map.find(param_name);
      if (vector_it != vector_map.end())
        for (unsigned int ii = 0; ii < std::get<1>(iter.second); ++ii)
          vec[std::get<0>(iter.second) + ii] = (vector_it->second)[ii];
    }
  }
}

void
DeepGaussianProcess::vecToMap(
    const std::unordered_map<std::string, std::tuple<unsigned int, unsigned int, Real, Real>> &
        tuning_data,
    std::unordered_map<std::string, Real> & scalar_map,
    std::unordered_map<std::string, std::vector<Real>> & vector_map,
    const std::vector<Real> & vec)
{
  for (auto iter : tuning_data)
  {
    const std::string & param_name = iter.first;
    if (scalar_map.find(param_name) != scalar_map.end())
      scalar_map[param_name] = vec[std::get<0>(iter.second)];
    else if (vector_map.find(param_name) != vector_map.end())
      for (unsigned int ii = 0; ii < std::get<1>(iter.second); ++ii)
        vector_map[param_name][ii] = vec[std::get<0>(iter.second) + ii];
  }
}

} // StochasticTools namespace

template <>
void
dataStore(std::ostream & stream, Eigen::LLT<RealEigenMatrix> & decomp, void * context)
{
  // Store the L matrix as opposed to the full matrix to avoid compounding
  // roundoff error and decomposition error
  RealEigenMatrix L(decomp.matrixL());
  dataStore(stream, L, context);
}

template <>
void
dataLoad(std::istream & stream, Eigen::LLT<RealEigenMatrix> & decomp, void * context)
{
  RealEigenMatrix L;
  dataLoad(stream, L, context);
  decomp.compute(L * L.transpose());
}

template <>
void
dataStore(std::ostream & stream, StochasticTools::DeepGaussianProcess & dgp_utils, void * context)
{
  dataStore(stream, dgp_utils.hyperparamMap(), context);
  dataStore(stream, dgp_utils.hyperparamVectorMap(), context);
  dataStore(stream, dgp_utils.covarType(), context);
  dataStore(stream, dgp_utils.covarName(), context);
  dataStore(stream, dgp_utils.covarNumOutputs(), context);
  dataStore(stream, dgp_utils.dependentCovarNames(), context);
  dataStore(stream, dgp_utils.dependentCovarTypes(), context);
  dataStore(stream, dgp_utils.K(), context);
  dataStore(stream, dgp_utils.KResultsSolve(), context);
  dataStore(stream, dgp_utils.KCholeskyDecomp(), context);
  dataStore(stream, dgp_utils.paramStandardizer(), context);
  dataStore(stream, dgp_utils.dataStandardizer(), context);
}

template <>
void
dataLoad(std::istream & stream, StochasticTools::DeepGaussianProcess & dgp_utils, void * context)
{
  dataLoad(stream, dgp_utils.hyperparamMap(), context);
  dataLoad(stream, dgp_utils.hyperparamVectorMap(), context);
  dataLoad(stream, dgp_utils.covarType(), context);
  dataLoad(stream, dgp_utils.covarName(), context);
  dataLoad(stream, dgp_utils.covarNumOutputs(), context);
  dataLoad(stream, dgp_utils.dependentCovarNames(), context);
  dataLoad(stream, dgp_utils.dependentCovarTypes(), context);
  dataLoad(stream, dgp_utils.K(), context);
  dataLoad(stream, dgp_utils.KResultsSolve(), context);
  dataLoad(stream, dgp_utils.KCholeskyDecomp(), context);
  dataLoad(stream, dgp_utils.paramStandardizer(), context);
  dataLoad(stream, dgp_utils.dataStandardizer(), context);
}
