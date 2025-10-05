// Source-based slice around line 432
// Method: <com.google.common.collect.Range: boolean test(C)>


  /**
   * @deprecated Provided only to satisfy the {@link java.util.function.Predicate} interface; use
   *     {@link #contains} instead.
   * @since 21.0
   */
  @InlineMe(replacement = "this.contains(input)")
  @Deprecated
  @Override
  public boolean test(C input) {
    return contains(input);
  }

  /**
   * Returns {@code true} if every element in {@code values} is {@linkplain #contains contained} in
   * this range.
   */
  public boolean containsAll(Iterable<? extends C> values) {
    if (Iterables.isEmpty(values)) {
      return true;
