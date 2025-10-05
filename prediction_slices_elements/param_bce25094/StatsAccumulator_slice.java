// Source-based slice around line 196
// Method: <com.google.common.math.StatsAccumulator: void merge(long,double,double,double,double)>

    }
    merge(values.count(), values.mean(), values.sumOfSquaresOfDeltas(), values.min(), values.max());
  }

  private void merge(
      long otherCount,
      double otherMean,
      double otherSumOfSquaresOfDeltas,
      double otherMin,
      double otherMax) {
    if (count == 0) {
      count = otherCount;
      mean = otherMean;
      sumOfSquaresOfDeltas = otherSumOfSquaresOfDeltas;
      min = otherMin;
      max = otherMax;
    } else {
      count += otherCount;
      if (isFinite(mean) && isFinite(otherMean)) {
        // This is a generalized version of the calculation in add(double) above.
