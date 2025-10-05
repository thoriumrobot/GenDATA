// Source-based slice around line 398
// Method: <com.google.common.math.Stats: double max()>

   *
   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
   * contains {@link Double#POSITIVE_INFINITY} and not {@link Double#NaN} then the result is {@link
   * Double#POSITIVE_INFINITY}. If it contains {@link Double#NEGATIVE_INFINITY} and finite values
   * only then the result is the highest finite value. If it contains {@link
   * Double#NEGATIVE_INFINITY} only then the result is {@link Double#NEGATIVE_INFINITY}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double max() {
    checkState(count != 0);
    return max;
  }

  /**
   * {@inheritDoc}
   *
   * <p><b>Note:</b> This tests exact equality of the calculated statistics, including the floating
   * point values. Two instances are guaranteed to be considered equal if one is copied from the
   * other using {@code second = new StatsAccumulator().addAll(first).snapshot()}, if both were
