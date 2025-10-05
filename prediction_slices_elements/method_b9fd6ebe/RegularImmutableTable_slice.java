// Source-based slice around line 118
// Method: <com.google.common.collect.RegularImmutableTable: RegularImmutableTable forCells(List,Comparator,Comparator)>

    @SuppressWarnings("RedundantOverride")
    @Override
    @J2ktIncompatible
    @GwtIncompatible
        Object writeReplace() {
      return super.writeReplace();
    }
  }

  static <R, C, V> RegularImmutableTable<R, C, V> forCells(
      List<Cell<R, C, V>> cells,
      @Nullable Comparator<? super R> rowComparator,
      @Nullable Comparator<? super C> columnComparator) {
    checkNotNull(cells);
    if (rowComparator != null || columnComparator != null) {
      /*
       * This sorting logic leads to a cellSet() ordering that may not be expected and that isn't
       * documented in the Javadoc. If a row Comparator is provided, cellSet() iterates across the
       * columns in the first row, the columns in the second row, etc. If a column Comparator is
       * provided but a row Comparator isn't, cellSet() iterates across the rows in the first
