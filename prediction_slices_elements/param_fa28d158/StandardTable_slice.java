// Source-based slice around line 313
// Method: <com.google.common.collect.StandardTable: Map row(R)>

            CollectSpliterators.map(
                rowEntry.getValue().entrySet().spliterator(),
                (Entry<C, V> columnEntry) ->
                    immutableCell(rowEntry.getKey(), columnEntry.getKey(), columnEntry.getValue())),
        Spliterator.DISTINCT | Spliterator.SIZED,
        size());
  }

  @Override
  public Map<C, V> row(R rowKey) {
    return new Row(rowKey);
  }

  class Row extends IteratorBasedAbstractMap<C, V> {
    final R rowKey;

    Row(R rowKey) {
      this.rowKey = checkNotNull(rowKey);
    }

