//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "SurrogateModel.h"
#include "Standardizer.h"
#include <Eigen/Dense>
#include "CovarianceInterface.h"
#include "DeepGaussianProcess.h"

class DeepGaussianProcessSurrogate : public SurrogateModel, public CovarianceInterface
{
public:
  static InputParameters validParams();
  DeepGaussianProcessSurrogate(const InputParameters & parameters);
  using SurrogateModel::evaluate;
  virtual Real evaluate(const std::vector<Real> & x) const;
  virtual void evaluate(const std::vector<Real> & x, std::vector<Real> & y) const;
  virtual Real evaluate(const std::vector<Real> & x, Real & std) const;
  virtual void
  evaluate(const std::vector<Real> & x, std::vector<Real> & y, std::vector<Real> & std) const;

  /**
   * This function is called by LoadCovarianceDataAction when the surrogate is
   * loading training data from a file. The action must recreate the covariance
   * object before this surrogate can set the correct pointer.
   */
  virtual void setupCovariance(UserObjectName _covar_name);

  StochasticTools::DeepGaussianProcess & dgp() { return _dgp; }
  const StochasticTools::DeepGaussianProcess & getDGP() const { return _dgp; }

private:
  StochasticTools::DeepGaussianProcess & _dgp;

  /// Paramaters (x) used for training
  const RealEigenMatrix & _training_params;
};
