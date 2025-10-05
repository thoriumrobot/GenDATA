// Source-based slice around line 36
// Method: <com.google.common.base.Internal: long toNanosSaturated(Duration)>

   * Returns the number of nanoseconds of the given duration without throwing or overflowing.
   *
   * <p>Instead of throwing {@link ArithmeticException}, this method silently saturates to either
   * {@link Long#MAX_VALUE} or {@link Long#MIN_VALUE}. This behavior can be useful when decomposing
   * a duration in order to call a legacy API which requires a {@code long, TimeUnit} pair.
   */
  // We use this method only for cases in which we need to decompose to primitives.
  @SuppressWarnings({"GoodTime-ApiWithNumericTimeUnit", "GoodTime-DecomposeToPrimitive"})
  @IgnoreJRERequirement
  static long toNanosSaturated(Duration duration) {
    // Using a try/catch seems lazy, but the catch block will rarely get invoked (except for
    // durations longer than approximately +/- 292 years).
    try {
      return duration.toNanos();
    } catch (ArithmeticException tooBig) {
      return duration.isNegative() ? Long.MIN_VALUE : Long.MAX_VALUE;
    }
  }

  private Internal() {}
