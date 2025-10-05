// Source-based slice around line 873
// Method: com.google.common.collect.StandardTable.columnMap

          return entry.getKey() != null
              && entry.getValue() instanceof Map
              && backingMap.entrySet().remove(entry);
        }
        return false;
      }
    }
  }

  @LazyInit private transient @Nullable ColumnMap columnMap;

  @Override
  public Map<C, Map<R, V>> columnMap() {
    ColumnMap result = columnMap;
    return (result == null) ? columnMap = new ColumnMap() : result;
  }

  @WeakOuter
  private final class ColumnMap extends ViewCachingAbstractMap<C, Map<R, V>> {
    // The cast to C occurs only when the key is in the map, implying that it
