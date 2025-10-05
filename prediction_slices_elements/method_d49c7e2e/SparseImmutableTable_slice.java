// Source-based slice around line 102
// Method: <com.google.common.collect.SparseImmutableTable: ImmutableMap rowMap()>

  }

  @Override
  public ImmutableMap<C, Map<R, V>> columnMap() {
    // Casts without copying.
    return ImmutableMap.copyOf(columnMap);
  }

  @Override
  public ImmutableMap<R, Map<C, V>> rowMap() {
    // Casts without copying.
    return ImmutableMap.copyOf(rowMap);
  }

  @Override
  public int size() {
    return cellRowIndices.length;
  }

  @Override
