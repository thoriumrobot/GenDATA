// Source-based slice around line 182
// Method: <com.google.common.collect.TreeBasedTable: SortedMap row(R)>

   *
   * <p>Because a {@code TreeBasedTable} has unique sorted values for a given row, this method
   * returns a {@link SortedMap}, instead of the {@link Map} specified in the {@link Table}
   * interface.
   *
   * @since 10.0 (<a href="https://github.com/google/guava/wiki/Compatibility" >mostly
   *     source-compatible</a> since 7.0)
   */
  @Override
  public SortedMap<C, V> row(R rowKey) {
    return new TreeRow(rowKey);
  }

  private final class TreeRow extends Row implements SortedMap<C, V> {
    final @Nullable C lowerBound;
    final @Nullable C upperBound;

    TreeRow(R rowKey) {
      this(rowKey, null, null);
    }
