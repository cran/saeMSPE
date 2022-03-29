\name{mspeFHlnr}
\alias{mspeFHlnr}
\title{
Compute MSPE through linearization method for Fay Herriot model
}
\description{
This function returns MSPE estimate with linearization appoximation method for Fay Herriot model.
}
\usage{
mspeFHlnr(Y, X, D, method = "PR", var.method = "default")
}

\arguments{
  \item{Y}{
a numeric vector. It represents the response value for Fay Herriot model.
}
  \item{X}{
a numeric matrix. It stands for the available auxiliary values.
}
  \item{D}{
a numeric vector consisting of the known sampling variances of each of the small area levels.
}
  \item{method}{
MSPE estimation method. See "Details".
}
  \item{var.method}{
Variance component estimation method. See "Details".
}
}
\details{
Default \code{method} for \code{mspeNERlnr} is "PR" ,proposed by N. G. N. Prasad and J. N. K. Rao, Prasad-Rao (PR) method uses Taylor series expansion to obtain a second-order approximation to the MSPE. Function \code{mspeNERlnr} also provide the following methods:

Method "DL" proposed by Datta and Lahiri , It advanced PR method to cover the cases when the variance components are estimated by ML and REML estimator. Set \code{method = "DL"}.

Method "DRS" proposed by Datta and Smith, It focus on the second order unbiasedness appoximation when the variance component is replaced by Empirical Bayes estimator. Set \code{method = "DRS"}.

Method "MPR" is a modified version of "PR", It was proposed by Liu et al. It is a robust method that broaden the mean function from the linear form. Set \code{method = "MPR"}.

Default \code{var.method} and available variance component estimation method for each method is list as follows:

For \code{method = "PR"}, \code{var.method = "MOM"} is the only available variance component estimation method,

For \code{method = "DL"}, \code{var.method = "ML"} or \code{var.method = "REML"} is available,

For \code{method = "DRS"}, \code{var.method = "EB"} is the only available variance component estimation method,

For \code{method = "MPR"}, \code{var.method = "OBP"} is the only available variance component estimation method.

}
\value{
This function returns a vector of the MSPE estimates.
}

\references{
N. G. N. Prasad and J. N. K. Rao. The estimation of the mean squared error of small-area estimators. \emph{Journal of the American Statistical Association}, 85(409):163-171, 1990.

G. S. Datta and P. Lahiri. A unified measure of uncertainty of estimated best linear unbiased predictors in small area estimation problems. \emph{Statistica Sinica}, 10(2):613-627, 2000.

G. S. Datta and R. D. D. Smith. On measuring the variability of small area estimators under a basic area level model. \emph{Biometrika}, 92(1):183-196, 2005.

X. Liu, H. Ma, and J. Jiang. That prasad-rao is robust: Estimation of mean squared prediction error of observed best predictor under potential model misspecification. \emph{Statistica Sinica}, 2020.
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
result = mspeFHlnr(Y,X,D,method = "PR", var.method = "default")
}

