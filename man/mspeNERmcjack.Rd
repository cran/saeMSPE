\name{mspeNERmcjack}
\alias{mspeNERmcjack}
\title{
Compute MSPE through resampling method for Nested error regression model
}
\description{
This function returns MSPE estimator with jackknife appoximation method for Nested error regression model
}
\usage{
mspeNERmcjack(ni, X, Y, Xmean, K = 50, method = 2)
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
  \item{Xmean}{
a numeric matrix. Stands for the population mean of auxiliary values.
}
  \item{K}{
a numeric vector. It represents the Monte-Carlo sample number for "mcjack", Default value is 50.
}
  \item{method}{
The MSPE estimation method to be used. See "Details".
}
}

\details{
This method was proposed by J. Jiang, P. Lahiri, and T. Nguyen, mcjack method uses Monte-Carlo approximation to ensure a strictly positive estimator.

Default value for \code{method} is 2, \code{method = 2} represents the REML method and available variance component estimation method for each method is list as follows:
\code{method = 1} represents MOM method,
\code{method = 3} represents ML method.

}

\value{
This function returns a vector of the MSPE estimates based on Monte-Carlo jackknife method.
}

\references{
M. H. Quenouille. Approximate tests of correlation in time series. \emph{Journal of the Royal Statistical Society. Series B (Methodological)}, 11(1):68-84, 1949.

J. W. Tukey. Bias and confidence in not quite large samples. \emph{Annals of Mathematical Statistics}, 29(2):614, 1958.

J. Jiang and L. S. M. Wan. A unified jackknife theory for empirical best prediction with m estimation. \emph{Annals of Statistics}, 30(6):1782-1810, 2002.

J. Jiang, P. Lahiri, and T. Nguyen. A unified monte carlo jackknife for small area estimation after model selection. \emph{Annals of Mathematical Sciences and Applications}, 3(2):405-438, 2018.
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
result = mspeNERmcjack(ni, X, Y, Xmean, method = 2, 10)
}

