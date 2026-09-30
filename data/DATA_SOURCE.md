# Data source

**Dataset:** MineThatData E-Mail Analytics and Data Mining Challenge (Kevin Hillstrom, 2008).
**Original announcement:** https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

64,000 customers who last purchased within the previous 12 months were randomly assigned to:

| `segment`       | Meaning                          | n      |
|-----------------|----------------------------------|--------|
| `Mens E-Mail`   | Received an email for men's merchandise   | 21,307 |
| `Womens E-Mail` | Received an email for women's merchandise | 21,387 |
| `No E-Mail`     | Control group, no email          | 21,306 |

Outcomes, measured over the following two weeks: `visit` (0/1), `conversion` (0/1), `spend` (dollars).

**File in this repo:** `hillstrom.csv.gz` is an unmodified copy of the file that
[`scikit-uplift`](https://www.uplift-modeling.com/) downloads with `fetch_hillstrom(target_col="all")`
(https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz). It is only 443 KB, so it is
committed to keep the tests and the deployed app independent of a third-party download.
`uplift.data.load_hillstrom()` falls back to downloading it if the file is missing.

The data was published openly for the challenge. All credit for the dataset goes to Kevin Hillstrom.
