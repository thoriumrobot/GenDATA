// Source-based slice around line 113
// Method: <com.google.common.collect.SparseImmutableTable: Cell getCell(int)>

    return ImmutableMap.copyOf(rowMap);
  }

  @Override
  public int size() {
    return cellRowIndices.length;
  }

  @Override
  Cell<R, C, V> getCell(int index) {
    int rowIndex = cellRowIndices[index];
    Entry<R, ImmutableMap<C, V>> rowEntry = rowMap.entrySet().asList().get(rowIndex);
    ImmutableMap<C, V> row = rowEntry.getValue();
    int columnIndex = cellColumnInRowIndices[index];
    Entry<C, V> colEntry = row.entrySet().asList().get(columnIndex);
    return cellOf(rowEntry.getKey(), colEntry.getKey(), colEntry.getValue());
  }

  @Override
  V getValue(int index) {
