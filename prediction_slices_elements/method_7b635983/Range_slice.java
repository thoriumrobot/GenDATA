// Source-based slice around line 420
// Method: <com.google.common.collect.Range: boolean apply(C)>

  }

  /**
   * @deprecated Provided only to satisfy the {@link Predicate} interface; use {@link #contains}
   *     instead.
   */
  @InlineMe(replacement = "this.contains(input)")
  @Deprecated
  @Override
  public boolean apply(C input) {
    return contains(input);
  }

  /**
   * @deprecated Provided only to satisfy the {@link java.util.function.Predicate} interface; use
   *     {@link #contains} instead.
   * @since 21.0
   */
  @InlineMe(replacement = "this.contains(input)")
  @Deprecated
