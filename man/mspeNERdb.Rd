\name{mspeNERdb}
\alias{mspeNERdb}
\title{
Compute MSPE through double bootstrap(DB) method for Nested error regression model
}
\description{
This function returns MSPE estimator with double bootstrap method for Nested error regression model.
}
\usage{
mspeNERdb(ni, X, Y, Xmean, K = 50, C = 50, method = 1)
}
\arguments{
  \item{ni}{
(vector). It represents the sample number for every small area.
}
  \item{X}{
(matrix).  It represents the small area response.
}
  \item{Y}{
(vector). It represents the design matrix.
}
  \item{Xmean}{
(matrix). Stands for the population mean of auxiliary values.
}
  \item{K}{
(integer). It represents the first bootstrap sample number. Default value is 50.
}
  \item{C}{
(integer). It represents the second bootstrap sample number. Default value is 50.
}
  \item{method}{
The variance component estimation method to be used. See "Details".
}
}
\details{
This method was proposed by P. Hall and T. Maiti. Double bootstrap method uses boostrap tool twice for NER model to avoid the unattractivitive bias correction: one is to estimate the estimator bias, the other is to correct for bias.

Default value for \code{method} is 1, \code{method = 1} represents the MOM method , \code{method = 2} and \code{method = 3} represents ML and REML method, respectively.
}
\value{
This function returns a list with components:
\item{MSPE}{(vector) MSPE estimates for NER model.}
\item{bhat}{(vector) Estimates of the unknown regression coefficients.}
\item{sigvhat2}{(numeric) Estimates of the area-specific variance component.}
\item{sigehat2}{(numeric) Estimates of the random error variance component.}
}
\references{
F. B. Butar and P. Lahiri. On measures of uncertainty of empirical bayes small area estimators. \emph{Journal of Statistical Planning and Inference}, 112(1-2):63-76, 2003.

N. G. N. Prasad and J. N. K. Rao. The estimation of the mean squared error of small-area estimators. \emph{Journal of the American Statistical Association}, 85(409):163-171, 1990.

Peter Hall and T. Maiti. On parametric bootstrap methods for small area prediction. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 2006a.

H. T. Maiti and T. Maiti. Nonparametric estimation of mean squared prediction error in nested error regression models. \emph{Annals of Statistics}, 34(4):1733-1750, 2006b.
}
\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
### parameter setting 
Ni = 1000; sigmaX = 1.5; m = 10
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
mspeNERdb(ni, X, Y, Xmean, 10, 10, method = 1)
}

