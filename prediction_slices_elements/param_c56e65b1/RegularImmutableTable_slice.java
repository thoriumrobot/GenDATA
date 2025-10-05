// Source-based slice around line 149
// Method: <com.google.common.collect.RegularImmutableTable: RegularImmutableTable forCells(Iterable)>

            return (columnComparator == null)
                ? 0
                : columnComparator.compare(cell1.getColumnKey(), cell2.getColumnKey());
          };
      sort(cells, comparator);
    }
    return forCellsInternal(cells, rowComparator, columnComparator);
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
