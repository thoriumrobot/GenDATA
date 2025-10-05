// Source-based slice around line 307
// Method: <com.google.common.collect.DiscreteDomain: long distance(C,C)>

   * {@link #previous} (if negative) are needed to reach {@code end} starting from {@code start}.
   * For example, if {@code end = next(next(next(start)))}, then {@code distance(start, end) == 3}
   * and {@code distance(end, start) == -3}. As well, {@code distance(a, a)} is always zero.
   *
   * <p>Note that this function is necessarily well-defined for any discrete type.
   *
   * @return the distance as described above, or {@link Long#MIN_VALUE} or {@link Long#MAX_VALUE} if
   *     the distance is too small or too large, respectively.
   */
  public abstract long distance(C start, C end);

  /**
   * Returns the minimum value of type {@code C}, if it has one. The minimum value is the unique
   * value for which {@link Comparable#compareTo(Object)} never returns a positive value for any
   * input of type {@code C}.
   *
   * <p>The default implementation throws {@code NoSuchElementException}.
   *
   * @return the minimum value of type {@code C}; never null
   * @throws NoSuchElementException if the type has no (practical) minimum value; for example,
