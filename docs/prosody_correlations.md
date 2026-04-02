# Prosody Feature Correlations with Aesthetic Score

Pearson and Spearman correlations between each of the 12 prosody features and the mean aesthetic score, computed across all 2,503 clips.

| Feature | Pearson r | p-value | Spearman rho | p-value |
|---------|-----------|---------|--------------|---------|
| pitch.mean | 0.2280 | <0.0001 | 0.1997 | <0.0001 |
| pitch.max | 0.1823 | <0.0001 | 0.1772 | <0.0001 |
| pitch.std | 0.1324 | <0.0001 | 0.1364 | <0.0001 |
| pitch.max_mean_ratio | -0.0342 | 0.0869 | -0.0095 | 0.6351 |
| energy.mean | 0.1274 | <0.0001 | 0.1580 | <0.0001 |
| energy.max | 0.1916 | <0.0001 | 0.2311 | <0.0001 |
| energy.std | 0.1759 | <0.0001 | 0.2209 | <0.0001 |
| energy.rate_of_change | -0.0614 | 0.0021 | -0.0380 | 0.0574 |
| speech_rate.chars_per_sec | -0.0548 | 0.0061 | -0.0678 | 0.0007 |
| speech_rate.words_per_sec | -0.0507 | 0.0112 | -0.0602 | 0.0026 |
| speech_rate.segment_count | 0.3051 | <0.0001 | 0.2399 | <0.0001 |
| spectral_centroid.mean | 0.0520 | 0.0093 | 0.0517 | 0.0097 |

## Summary

The strongest correlate is `speech_rate.segment_count` (Spearman rho = 0.24), which reflects how many distinct commentary segments appear in a clip. Clips rated higher tend to have more commentary activity. Pitch and energy features also show positive correlations (rho in the 0.14-0.23 range), indicating that commentator vocal intensity tracks with perceived clip quality.

Speech rate (words or characters per second) shows a weak negative correlation, suggesting that faster speech does not predict higher-rated clips. The pitch max-to-mean ratio and energy rate of change are not significant at p < 0.01.
