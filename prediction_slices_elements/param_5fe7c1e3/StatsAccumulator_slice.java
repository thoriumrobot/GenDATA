// Source-based slice around line 195
// Method: <com.google.common.math.StatsAccumulator: void merge(long,double,double,double,double)>

      return;
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
