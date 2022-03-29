\name{mspeFHsumca}
\alias{mspeFHsumca}
\title{
Compute MSPE through Sumca method for Fay Herriot model
}
\description{
This function returns MSPE estimator with the combination of linearization and resampling appoximation method called "Sumca", for Fay Herriot model.
}
\usage{
mspeFHsumca(Y, X, D, K = 50, method = 2)
}

\arguments{
  \item{Y}{
a numeric vector. It represents the response value for Fay Herriot model.
}
  \item{X}{
a numeric matrix. Stands for the available auxiliary values.
}
  \item{D}{
a numeric vector. It represents the knowing sampling variance for Fay Herriot model.
}
  \item{K}{
a numeric vector. It represents the Monte-Carlo sample size for "sumca". Default value is 50.
}
  \item{method}{
It represents the variance component estimation method. See "Details".
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
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
result = mspeFHsumca(Y,X,D,K = 50)
}

