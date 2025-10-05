// Source-based slice around line 407
// Method: <com.google.common.collect.Range: boolean contains(C)>

  public boolean isEmpty() {
    return lowerBound.equals(upperBound);
  }

  /**
   * Returns {@code true} if {@code value} is within the bounds of this range. For example, on the
   * range {@code [0..2)}, {@code contains(1)} returns {@code true}, while {@code contains(2)}
   * returns {@code false}.
   */
  public boolean contains(C value) {
    checkNotNull(value);
    // let this throw CCE if there is some trickery going on
    return lowerBound.isLessThan(value) && !upperBound.isLessThan(value);
  }

  /**
   * @deprecated Provided only to satisfy the {@link Predicate} interface; use {@link #contains}
   *     instead.
   */
  @InlineMe(replacement = "this.contains(input)")
