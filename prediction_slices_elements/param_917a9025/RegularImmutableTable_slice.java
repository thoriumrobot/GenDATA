// Source-based slice around line 156
// Method: <com.google.common.collect.RegularImmutableTable: RegularImmutableTable forCellsInternal(Iterable,Comparator,Comparator)>

  }

  static <R, C, V> RegularImmutableTable<R, C, V> forCells(Iterable<Cell<R, C, V>> cells) {
    return forCellsInternal(cells, null, null);
  }

  private static <R, C, V> RegularImmutableTable<R, C, V> forCellsInternal(
      Iterable<Cell<R, C, V>> cells,
      @Nullable Comparator<? super R> rowComparator,
      @Nullable Comparator<? super C> columnComparator) {
    Set<R> rowSpaceBuilder = new LinkedHashSet<>();
    Set<C> columnSpaceBuilder = new LinkedHashSet<>();
    ImmutableList<Cell<R, C, V>> cellList = ImmutableList.copyOf(cells);
    for (Cell<R, C, V> cell : cells) {
      rowSpaceBuilder.add(cell.getRowKey());
      columnSpaceBuilder.add(cell.getColumnKey());
    }

    ImmutableSet<R> rowSpace =
        (rowComparator == null)
