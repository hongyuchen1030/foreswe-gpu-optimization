# Figure 4(a) Station-Mean Aggregation Comparison

This diagnostic plot keeps the paper-style grouping and target-date month assignment fixed,
but changes the boxplot samples from pooled station-day bias values to station-level monthly mean bias values.

Settings:
- Groups: paper-style averaged max SWE, 30/60/90 percentile thresholds.
- Month assignment: target date = origin date + horizon days.
- Horizons: [1, 3, 5, 7, 10].
- Relative bias: `100 * (predicted - actual) / actual`.
- Minimum actual SWE threshold: > 1.0 mm.

## April Group 1 Comparison

Per-year comparison between pooled station-day aggregation and station-level mean aggregation:

```text
 year  horizon  pooled_n_samples  pooled_mean_bias  pooled_median_bias  station_n  station_mean_bias  station_median_bias
 2014        1              1766         18.938680           -4.942796        112          35.041307             5.669660
 2014        3              1766         33.725955            1.151454        112          41.758273            17.337046
 2014        5              1766         51.594429            2.513034        112          63.642063            28.978828
 2014        7              1766         86.169860           12.507725        112         111.374146            57.289093
 2014       10              1766        114.324173           18.904621        112         148.804476            80.673169
 2015        1              2490         -5.830867          -10.697958        125          -2.602582            -8.423293
 2015        3              2490         15.745528           -1.706821        125          23.103910             9.479613
 2015        5              2490         27.358456           -1.711988        125          34.078254            16.059076
 2015        7              2490         52.239466            4.722614        125          66.816360            31.915148
 2015       10              2490         73.304379            7.232340        125          95.374421            49.300883
 2016        1              2966          3.645559           -5.528367        130          18.306536            -3.354480
 2016        3              2966         18.071122           -0.446659        130          32.962645             4.826063
 2016        5              2966         30.200727            0.010361        130          52.290852            10.496812
 2016        7              2966         49.660854            3.918234        130          81.987676            20.523550
 2016       10              2966         69.789318            6.354135        130         108.829170            33.767073
 2017        1              2594         -6.012840           -7.547532        119          -2.500206            -5.629045
 2017        3              2594         15.290375           -3.045433        119          25.567055             4.297335
 2017        5              2594         22.358974           -5.379921        119          40.291918             7.396731
 2017        7              2594         46.591682           -2.801123        119          85.237527            21.240832
 2017       10              2594         54.762376           -4.710118        119         100.931438            24.181973
 2018        1              3556         -6.588200           -8.239882        142          -5.693333            -6.939994
 2018        3              3556         16.636167           -1.163968        142          28.196815            12.058931
 2018        5              3556         28.659093           -1.495606        142          49.424217            21.786246
 2018        7              3556         53.728434            3.115958        142          92.211016            38.448648
 2018       10              3556         70.224060            2.802632        142         115.832931            47.690973
```

Average across the five released test seasons:

```text
 horizon  pooled_mean_bias  pooled_median_bias  station_mean_bias  station_median_bias
       1          0.830467           -7.391307           8.510344            -3.735430
       3         19.893829           -1.042285          30.317740             9.599798
       5         32.034336           -1.212824          47.945461            16.943538
       7         57.678059            4.292681          87.525345            33.883454
      10         76.480861            6.116722         113.954487            47.122814
```

## Interpretation

- At 3 days, the average pooled median bias is -1.042, while the average station-level median bias is 9.600.
- At 5 days, the average pooled median bias is -1.213, while the average station-level median bias is 16.944.
- At 1 day, the average pooled median bias is -7.391, and the average station-level median bias is -3.735.
- This means station-level mean aggregation does move April Group 1 upward, especially at 3-5 days.
- It does not fully flip the 1-day April Group 1 bias positive, so it is a plausible partial explanation rather than a complete resolution on its own.

## Outputs

- PNG: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_station_mean_aggregation.png`
- PDF: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_station_mean_aggregation.pdf`
- CSV: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_bias_summary_station_mean_aggregation.csv`
