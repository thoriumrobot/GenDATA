// Source-based slice around line 294
// Method: <com.google.common.collect.DiscreteDomain: C previous(C)>


  /**
   * Returns the unique greatest value of type {@code C} that is less than {@code value}, or {@code
   * null} if none exists. Inverse operation to {@link #next}.
   *
   * @param value any value of type {@code C}
   * @return the greatest value less than {@code value}, or {@code null} if {@code value} is {@code
   *     minValue()}
   */
  public abstract @Nullable C previous(C value);

  /**
   * Returns a signed value indicating how many nested invocations of {@link #next} (if positive) or
   * {@link #previous} (if negative) are needed to reach {@code end} starting from {@code start}.
   * For example, if {@code end = next(next(next(start)))}, then {@code distance(start, end) == 3}
   * and {@code distance(end, start) == -3}. As well, {@code distance(a, a)} is always zero.
   *
   * <p>Note that this function is necessarily well-defined for any discrete type.
   *
   * @return the distance as described above, or {@link Long#MIN_VALUE} or {@link Long#MAX_VALUE} if
