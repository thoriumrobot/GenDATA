// Source-based slice around line 398
// Method: <com.google.common.collect.Range: boolean isEmpty()>

  /**
   * Returns {@code true} if this range is of the form {@code [v..v)} or {@code (v..v]}. (This does
   * not encompass ranges of the form {@code (v..v)}, because such ranges are <i>invalid</i> and
   * can't be constructed at all.)
   *
   * <p>Note that certain discrete ranges such as the integer range {@code (3..4)} are <b>not</b>
   * considered empty, even though they contain no actual values. In these cases, it may be helpful
   * to preprocess ranges with {@link #canonical(DiscreteDomain)}.
   */
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
