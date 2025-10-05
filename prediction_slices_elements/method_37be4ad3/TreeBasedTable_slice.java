// Source-based slice around line 148
// Method: <com.google.common.collect.TreeBasedTable: Comparator rowComparator()>

   * Returns the comparator that orders the rows. With natural ordering, {@link Ordering#natural()}
   * is returned.
   *
   * @deprecated Use {@code table.rowKeySet().comparator()} instead.
   */
  @InlineMe(
      replacement = "requireNonNull(this.rowKeySet().comparator())",
      staticImports = "java.util.Objects.requireNonNull")
  @Deprecated
  public final Comparator<? super R> rowComparator() {
    /*
     * requireNonNull is safe because the factories require non-null Comparators, which they pass on
     * to the backing collections.
     */
    return requireNonNull(rowKeySet().comparator());
  }

  /**
   * Returns the comparator that orders the columns. With natural ordering, {@link
   * Ordering#natural()} is returned.
