// Source-based slice around line 96
// Method: <com.google.common.collect.SparseImmutableTable: ImmutableMap columnMap()>

    ImmutableMap.Builder<C, ImmutableMap<R, V>> columnBuilder =
        new ImmutableMap.Builder<>(columns.size());
    for (Entry<C, Map<R, V>> col : columns.entrySet()) {
      columnBuilder.put(col.getKey(), ImmutableMap.copyOf(col.getValue()));
    }
    this.columnMap = columnBuilder.buildOrThrow();
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

