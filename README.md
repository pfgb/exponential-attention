# exponential-attention
Code for a new Deep Learning model for the classification of time series used to classify ECGs.


The model calculates the Generalized Hurst Exponent on windows of a time series and uses this as an input for a self-attention mechanism.

The result of the self-attention mechanism is then concatenated to the entrance of every block of a N-BEATS model.
