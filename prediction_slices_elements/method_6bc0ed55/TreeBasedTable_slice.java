// Source-based slice around line 310
// Method: <com.google.common.collect.TreeBasedTable: Iterator createColumnKeyIterator()>

    @Override
    public @Nullable V put(C key, V value) {
      checkArgument(rangeContains(checkNotNull(key)));
      return super.put(key, value);
    }
  }

  /** Overridden column iterator to return columns values in globally sorted order. */
  @Override
  Iterator<C> createColumnKeyIterator() {
    Comparator<? super C> comparator = columnComparator();

    Iterator<C> merged =
        mergeSorted(
            transform(backingMap.values(), (Map<C, V> input) -> input.keySet().iterator()),
            comparator);

    return new AbstractIterator<C>() {
      @Nullable C lastValue;

