// Source-based slice around line 180
// Method: <com.google.common.collect.RegularImmutableTable: RegularImmutableTable forOrderedComponents(ImmutableList,ImmutableSet,ImmutableSet)>

            ? ImmutableSet.copyOf(columnSpaceBuilder)
            : ImmutableSet.copyOf(ImmutableList.sortedCopyOf(columnComparator, columnSpaceBuilder));

    return forOrderedComponents(cellList, rowSpace, columnSpace);
  }

  /** A factory that chooses the most space-efficient representation of the table. */
  static <R, C, V> RegularImmutableTable<R, C, V> forOrderedComponents(
      ImmutableList<Cell<R, C, V>> cellList,
      ImmutableSet<R> rowSpace,
      ImmutableSet<C> columnSpace) {
    // use a dense table if more than half of the cells have values
    // TODO(gak): tune this condition based on empirical evidence
    return (cellList.size() > (((long) rowSpace.size() * columnSpace.size()) / 2))
        ? new DenseImmutableTable<R, C, V>(cellList, rowSpace, columnSpace)
        : new SparseImmutableTable<R, C, V>(cellList, rowSpace, columnSpace);
  }

  /**
   * @throws IllegalArgumentException if {@code existingValue} is not null.
