// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// varfh
double varfh(Rcpp::Formula formula, Rcpp::DataFrame data, arma::vec D, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_varfh(SEXP formulaSEXP, SEXP dataSEXP, SEXP DSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(varfh(formula, data, D, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeFHjack
List mspeFHjack(Rcpp::Formula formula, Rcpp::DataFrame data, arma::vec D, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeFHjack(SEXP formulaSEXP, SEXP dataSEXP, SEXP DSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeFHjack(formula, data, D, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeFHpb
List mspeFHpb(Rcpp::Formula formula, Rcpp::DataFrame data, arma::vec D, int K, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeFHpb(SEXP formulaSEXP, SEXP dataSEXP, SEXP DSEXP, SEXP KSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeFHpb(formula, data, D, K, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeFHdb
List mspeFHdb(Rcpp::Formula formula, Rcpp::DataFrame data, arma::vec D, int K, int C, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeFHdb(SEXP formulaSEXP, SEXP dataSEXP, SEXP DSEXP, SEXP KSEXP, SEXP CSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type C(CSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeFHdb(formula, data, D, K, C, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeFHsumca
List mspeFHsumca(Rcpp::Formula formula, Rcpp::DataFrame data, arma::vec D, int K, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeFHsumca(SEXP formulaSEXP, SEXP dataSEXP, SEXP DSEXP, SEXP KSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeFHsumca(formula, data, D, K, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// varner
List varner(arma::vec ni, Rcpp::Formula formula, Rcpp::DataFrame data, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_varner(SEXP niSEXP, SEXP formulaSEXP, SEXP dataSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type ni(niSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(varner(ni, formula, data, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeNERjack
List mspeNERjack(arma::vec ni, Rcpp::Formula formula, Rcpp::DataFrame data, arma::mat Xmean, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeNERjack(SEXP niSEXP, SEXP formulaSEXP, SEXP dataSEXP, SEXP XmeanSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type ni(niSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Xmean(XmeanSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeNERjack(ni, formula, data, Xmean, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeNERpb
List mspeNERpb(arma::vec ni, Rcpp::Formula formula, Rcpp::DataFrame data, arma::mat Xmean, int K, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeNERpb(SEXP niSEXP, SEXP formulaSEXP, SEXP dataSEXP, SEXP XmeanSEXP, SEXP KSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type ni(niSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Xmean(XmeanSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeNERpb(ni, formula, data, Xmean, K, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeNERdb
List mspeNERdb(arma::vec ni, Rcpp::Formula formula, Rcpp::DataFrame data, arma::mat Xmean, int K, int C, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeNERdb(SEXP niSEXP, SEXP formulaSEXP, SEXP dataSEXP, SEXP XmeanSEXP, SEXP KSEXP, SEXP CSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type ni(niSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Xmean(XmeanSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type C(CSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeNERdb(ni, formula, data, Xmean, K, C, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}
// mspeNERsumca
List mspeNERsumca(arma::vec ni, Rcpp::Formula formula, Rcpp::DataFrame data, arma::mat Xmean, int K, int method, bool na_rm, bool na_omit);
RcppExport SEXP _saeMSPE_mspeNERsumca(SEXP niSEXP, SEXP formulaSEXP, SEXP dataSEXP, SEXP XmeanSEXP, SEXP KSEXP, SEXP methodSEXP, SEXP na_rmSEXP, SEXP na_omitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type ni(niSEXP);
    Rcpp::traits::input_parameter< Rcpp::Formula >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Xmean(XmeanSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< bool >::type na_rm(na_rmSEXP);
    Rcpp::traits::input_parameter< bool >::type na_omit(na_omitSEXP);
    rcpp_result_gen = Rcpp::wrap(mspeNERsumca(ni, formula, data, Xmean, K, method, na_rm, na_omit));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_saeMSPE_varfh", (DL_FUNC) &_saeMSPE_varfh, 6},
    {"_saeMSPE_mspeFHjack", (DL_FUNC) &_saeMSPE_mspeFHjack, 6},
    {"_saeMSPE_mspeFHpb", (DL_FUNC) &_saeMSPE_mspeFHpb, 7},
    {"_saeMSPE_mspeFHdb", (DL_FUNC) &_saeMSPE_mspeFHdb, 8},
    {"_saeMSPE_mspeFHsumca", (DL_FUNC) &_saeMSPE_mspeFHsumca, 7},
    {"_saeMSPE_varner", (DL_FUNC) &_saeMSPE_varner, 6},
    {"_saeMSPE_mspeNERjack", (DL_FUNC) &_saeMSPE_mspeNERjack, 7},
    {"_saeMSPE_mspeNERpb", (DL_FUNC) &_saeMSPE_mspeNERpb, 8},
    {"_saeMSPE_mspeNERdb", (DL_FUNC) &_saeMSPE_mspeNERdb, 9},
    {"_saeMSPE_mspeNERsumca", (DL_FUNC) &_saeMSPE_mspeNERsumca, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_saeMSPE(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
