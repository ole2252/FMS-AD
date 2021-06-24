In this folder 2 data types are present:
1. Decentralized system data (.pkl files)
2. Non-decentralized system data (.csv files)

The decentralized system data is time-series data, where the columns represent the available resource related metrics. The columns also include a timestamp column, label column that represents of the observation is anomalous or not and a txn_failure_num which is the number of transaction failures. Every row represents a single observation.

For the alternative data set, also time-series data is available. There are five resource related metrics available in the columns besides that the completion status is available as a column.
