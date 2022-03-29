\name{mspeNERsumca}
\alias{mspeNERsumca}
\title{
Compute MSPE through Sumca method for Nested error regression model
}
\description{
This function returns MSPE estimator with the combination of linearization and resampling appoximation method for Nested error regression model.
}
\usage{
mspeNERsumca(ni, X, Y, Xmean, K = 50, method = 2)
}

\arguments{
  \item{ni}{
a numeric vector. It represents the sample number for every small area.
}
  \item{X}{
a numeric matrix.  It represents the small area response.
}
  \item{Y}{
a numeric vector. It represents the design matrix.
}
  \item{Xmean}{
a numeric matrix. Stands for the population mean of auxiliary values.
}
  \item{K}{
a numeric vector. It represents the Monte-Carlo sample number for "sumca". Default value is 50.
}
  \item{method}{
The MSPE estimation method to be used. See "Details".
}
}

\details{
This method was proposed by J. Jiang, P. Lahiri, and T. Nguyen, sumca method combines the advantages of linearization and resampling methods and obtains unified, positive, low-computation burden and second-order unbiased MSPE estimators.

Default value for \code{method} is 2, \code{method = 2} represents the REML method and \code{method = 1} represents MOM method.

}

\value{
This function returns a vector of the MSPE estimates based on Sumca method.
}

\references{
J. Jiang and M. Torabi. Sumca: simple; unified; monte carlo assisted approach to second order unbiased mean squared prediction error estimation. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 82(2):467-485, 2020.
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
result = mspeNERsumca(ni, X, Y, Xmean, method = 2, 50)
}

