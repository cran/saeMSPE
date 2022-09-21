\name{mspeFHjack}
\alias{mspeFHjack}
\title{
Compute MSPE through Jackknife-based MSPE estimation method for Fay Herriot model
}
\description{
This function returns MSPE estimator with jackknife method for Fay Herriot model.
}
\usage{
mspeFHjack(Y, X, D, method = 1)
}

\arguments{
  \item{Y}{
(vector). It represents the response value for Fay Herriot model.
}
  \item{X}{
(matrix). It stands for the available auxiliary values.
}
  \item{D}{
(vector). Stands for the known sampling variances of each small area levels.
}
  \item{method}{
The variance component estimation method to be used. See "Details".
}

}

\details{
This bias-corrected jackknife MSPE estimator was proposed by J. Jiang and L. S. M. Wan, it covers a fairly general class of mixed models which includes gLMM, mixed logistic model and some of the widely used mixed linear models as special cases.

Default value for \code{method} is 1, \code{method = 1} represents the MOM method , \code{method = 2} and \code{method = 3} represents ML and REML method, respectively.

}

\value{
This function returns a list with components:
\item{MSPE}{(vector) MSPE estimates for Fay Herriot model.}
\item{bhat}{(vector) Estimates of the unknown regression coefficients.}
\item{Ahat}{(numeric) Estimates of the variance component.}
}

\references{
M. H. Quenouille. Approximate tests of correlation in time series. \emph{Journal of the Royal Statistical Society. Series B (Methodological)}, 11(1):68-84, 1949.

J. W. Tukey. Bias and confidence in not quite large samples. \emph{Annals of Mathematical Statistics}, 29(2):614, 1958.

J. Jiang and L. S. M. Wan. A unified jackknife theory for empirical best prediction with m estimation. \emph{Annals of Statistics}, 30(6):1782-1810, 2002.

}

\author{
Peiwen Xiao, Xiaohui Liu, Yuzi Liu, Jiming Jiang, and Shaochu Liu
}


\examples{
X = matrix(runif(10 * 3), 10, 3)
X[,1] = rep(1, 10) 
D = (1:10) / 10 + 0.5
Y = X \%*\% c(0.5,1,1.5) + rnorm(10, 0, sqrt(2)) + rnorm(10, 0, sqrt(D))
mspeFHjack(Y, X, D, method = 1)
}

