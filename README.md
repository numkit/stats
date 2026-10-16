# numkit/stats

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Statistics and Machine Learning Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance probability distributions, descriptive statistics, hypothesis testing, regression, clustering, and resampling algorithms with Highway SIMD vectorization and MATLAB R2025b parity.

---

## Features

- **Descriptive & NaN-Aware Statistics**: `var`, `std`, `median`, `quantile`, `prctile`, `mode`, `cov`, `corrcoef`, `skewness`, `kurtosis`, `iqr`, `mad`, `zscore`, `normalize`, `nansum`, `nanmean`, `nanstd`, `nanmedian`, `ksdensity`, `ecdf`.
- **Probability Distributions**: Continuous, discrete, and multivariate distributions (`norm*`, `t*`, `chi2*`, `f*`, `gam*`, `beta*`, `wbl*`, `poiss*`, `bino*`, `hyge*`, `nbin*`, `mvn*`, `mvt*`, `wishrnd`, `copula*`) plus generic `pdf`, `cdf`, `icdf`, `random`, `mle`, and `fitdist`.
- **Hypothesis Testing & ANOVA**: `ttest`, `ttest2`, `ztest`, `vartest`, `kstest`, `kstest2`, `lillietest`, `jbtest`, `adtest`, `dwtest`, `signrank`, `ranksum`, `chi2gof`, `fishertest`, `anova1`, `anova2`, `kruskalwallis`, `friedman`, `multcompare`.
- **Regression & Dimensionality Reduction**: `regress`, `nlinfit`, `nlparci`, `nlpredci`, `robustfit`, `glmfit`, `glmval`, `lasso`, `lassoglm`, `ridge`, `lscov`, `pca`, `pcacov`, `pcares`.
- **Clustering & Classification**: `kmeans`, `kmedoids`, `dbscan`, `linkage`, `cluster`, `pdist`, `pdist2`, `silhouette`, `knnsearch`, `rangesearch`, `classify`.
- **Resampling & Quasi-Monte Carlo**: `bootstrp`, `bootci`, `jackknife`, `crossval`, `randsample`, `datasample`, `lhsdesign`, `lhsnorm`, `haltonset`.

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `numkit_stats.dll` (Windows) or `libnumkit_stats.so` (Linux/macOS).

---

## Usage in NumKit

Install via the NumKit package manager:

```bash
nk pkg install stats
```

Or load dynamically in a script:

```matlab
% Loaded automatically on first call to any stats function:
y = normpdf(0, 0, 1);
idx = kmeans([1 1; 1.5 2; 3 4; 5 7; 3.5 5; 4.5 5; 3.5 4.5], 2);
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
