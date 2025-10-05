// Source-based slice around line 44
// Method: com.google.common.collect.SparseImmutableTable.cellColumnInRowIndices

  private final ImmutableMap<C, ImmutableMap<R, V>> columnMap;

  // For each cell in iteration order, the index of that cell's row key in the row key list.
  @SuppressWarnings("Immutable") // We don't modify this after construction.
  private final int[] cellRowIndices;

  // For each cell in iteration order, the index of that cell's column key in the list of column
  // keys present in that row.
  @SuppressWarnings("Immutable") // We don't modify this after construction.
  private final int[] cellColumnInRowIndices;

  SparseImmutableTable(
      ImmutableList<Cell<R, C, V>> cellList,
      ImmutableSet<R> rowSpace,
      ImmutableSet<C> columnSpace) {
    Map<R, Integer> rowIndex = Maps.indexMap(rowSpace);
    Map<R, Map<C, V>> rows = new LinkedHashMap<>();
    for (R row : rowSpace) {
      rows.put(row, new LinkedHashMap<C, V>());
    }
