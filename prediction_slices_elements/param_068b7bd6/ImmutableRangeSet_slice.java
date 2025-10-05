// Source-based slice around line 265
// Method: <com.google.common.collect.ImmutableRangeSet: void addAll(Iterable)>

  /**
   * Guaranteed to throw an exception and leave the {@code RangeSet} unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @Deprecated
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  public void addAll(Iterable<Range<C>> other) {
    throw new UnsupportedOperationException();
  }

  /**
   * Guaranteed to throw an exception and leave the {@code RangeSet} unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @Deprecated
