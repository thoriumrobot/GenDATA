// Source-based slice around line 760
// Method: <com.google.common.collect.StandardTable: Iterator createColumnKeyIterator()>

    }

    @Override
    public boolean contains(@Nullable Object obj) {
      return containsColumn(obj);
    }
  }

  /** Creates an iterator that returns each column value with duplicates omitted. */
  Iterator<C> createColumnKeyIterator() {
    return new ColumnKeyIterator();
  }

  private final class ColumnKeyIterator extends AbstractIterator<C> {
    // Use the same map type to support TreeMaps with comparators that aren't
    // consistent with equals().
    final Map<C, V> seen = factory.get();
    final Iterator<Map<C, V>> mapIterator = backingMap.values().iterator();
    Iterator<Entry<C, V>> entryIterator = emptyIterator();

