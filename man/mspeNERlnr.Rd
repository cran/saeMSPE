\name{mspeNERlnr}
\alias{mspeNERlnr}
\title{
Compute MSPE through linearization method for Nested error regression model
}
\description{
This function returns MSPE estimator with linearization appoximation method for Nested error regression model.
}
\usage{
mspeNERlnr(ni, X, Y, X.mean, method = "PR", var.method = "default")
}

\arguments{
  \item{ni}{
a numeric vector. It represents the sample number for every small area.
}
  \item{X}{
a numeric matrix. Stands for the available auxiliary values.
}
  \item{Y}{
a numeric vector. It represents the response value for Nested error regression model.
}
  \item{X.mean}{
a numeric matrix. Stands for the population mean of auxiliary values.
}
  \item{method}{
The MSPE estimation method to be used. See "Details".
}
  \item{var.method}{
The variance component estimation method to be used. See "Details".
}
}
\details{
Default \code{method} for \code{mspeNERlnr} is "PR" ,proposed by N. G. N. Prasad and J. N. K. Rao, Prasad-Rao (PR) method uses Taylor series expansion to obtain a second-order approximation to
the MSPE. Function \code{mspeNERlnr} also provide the following method:

Method "DL" advanced PR method to cover the cases when the variance components are estimated by ML and REML estimator. Set \code{method = "DL"}.

For \code{method = "PR"}, \code{var.method = "MOM"} is the only available variance component estimation method,

For \code{method = "DL"}, \code{var.method = "ML"} or \code{var.method = "REML"} are available.

}

\value{
This function returns a vector of the MSPE estimates.
}

\references{
N. G. N. Prasad and J. N. K. Rao. The estimation of the mean squared error of small-area estimators. \emph{Journal of the American Statistical Association}, 85(409):163-171, 1990.

G. S. Datta and P. Lahiri. A unified measure of uncertainty of estimated best linear unbiased predictors in small area estimation problems. \emph{Statistica Sinica}, 10(2):613-627, 2000.
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
### parameter setting 
Ni = 1000; sigmaX = 1.5; K = 100; C = 50; m = 10
beta = c(0.5, 1)
sigma_v2 = 0.8; sigma_e2 = 1
ni = sample(seq(1,10), m,replace = TRUE); n = sum(ni)
p = length(beta)
### population function
pop.model = function(Ni, sigmaX, beta, sigma_v2, sigma_e2, m){
  x = rnorm(m * Ni, 1, sqrt(sigmaX)); v = rnorm(m, 0, sqrt(sigma_v2)); y = numeric(m * Ni)
  theta = numeric(m); kk = 1
  for(i in 1 : m){
    sumx = 0
    for(j in 1:Ni){
      sumx = sumx + x[kk]
      y[kk] = beta[1] + beta[2] * x[kk] + v[i] + rnorm(1, 0, sqrt(sigma_e2))
      kk = kk + 1
    }
    meanx = sumx/Ni
    theta[i] = beta[1] + beta[2] * meanx + v[i]
  }
  group = rep(seq(m), each = Ni)
  x = cbind(rep(1, m*Ni), x)
  data = cbind(x, y, group)
  return(list(data = data, theta = theta))
} 
### sample function
sampleXY = function(Ni, ni, m, Population){
  Indx = c()
  for(i in 1:m){
    Indx = c(Indx, sample(c(((i - 1) * Ni + 1) : (i * Ni)), ni[i]))
  }
  Sample = Population[Indx, ]; Nonsample = Population[-Indx, ]
  return(list(Sample, Nonsample))
} 
### data generation process
Population = pop.model(Ni, sigmaX, beta, sigma_v2, sigma_e2, m)$data
XY = sampleXY(Ni, ni, m, Population)[[1]]
X = XY[, 1:p]
Y = XY[, p+1]
Xmean = matrix(NA, m, p)
for(tt in 1: m){
  Xmean[tt, ] = colMeans(Population[which(Population[,p+2] == tt), 1:p])
}
### mspe result
result = mspeNERlnr(ni, X, Y, Xmean, method = "PR", var.method = "default")
}

