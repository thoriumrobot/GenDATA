// Source-based slice around line 796
// Method: <com.google.common.collect.StandardTable: Collection values()>

  }

  /**
   * {@inheritDoc}
   *
   * <p>The collection's iterator traverses the values for the first row, the values for the second
   * row, and so on.
   */
  @Override
  public Collection<V> values() {
    return super.values();
  }

  @LazyInit private transient @Nullable Map<R, Map<C, V>> rowMap;

  @Override
  public Map<R, Map<C, V>> rowMap() {
    Map<R, Map<C, V>> result = rowMap;
    return (result == null) ? rowMap = createRowMap() : result;
  }
