// Source-based slice around line 300
// Method: <com.google.common.collect.StandardTable: Spliterator cellSpliterator()>

       */
      if (requireNonNull(rowEntry).getValue().isEmpty()) {
        rowIterator.remove();
        rowEntry = null;
      }
    }
  }

  @Override
  Spliterator<Cell<R, C, V>> cellSpliterator() {
    return CollectSpliterators.flatMap(
        backingMap.entrySet().spliterator(),
        (Entry<R, Map<C, V>> rowEntry) ->
            CollectSpliterators.map(
                rowEntry.getValue().entrySet().spliterator(),
                (Entry<C, V> columnEntry) ->
                    immutableCell(rowEntry.getKey(), columnEntry.getKey(), columnEntry.getValue())),
        Spliterator.DISTINCT | Spliterator.SIZED,
        size());
  }
