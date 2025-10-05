// Source-based slice around line 81
// Method: <com.google.common.collect.HashBasedTable: HashBasedTable create(int,int)>


  /**
   * Creates an empty {@code HashBasedTable} with the specified map sizes.
   *
   * @param expectedRows the expected number of distinct row keys
   * @param expectedCellsPerRow the expected number of column key / value mappings in each row
   * @throws IllegalArgumentException if {@code expectedRows} or {@code expectedCellsPerRow} is
   *     negative
   */
  public static <R, C, V> HashBasedTable<R, C, V> create(
      int expectedRows, int expectedCellsPerRow) {
    checkNonnegative(expectedCellsPerRow, "expectedCellsPerRow");
    Map<R, Map<C, V>> backingMap = Maps.newLinkedHashMapWithExpectedSize(expectedRows);
    return new HashBasedTable<>(backingMap, new Factory<C, V>(expectedCellsPerRow));
  }

  /**
   * Creates a {@code HashBasedTable} with the same mappings as the specified table.
   *
   * @param table the table to copy
