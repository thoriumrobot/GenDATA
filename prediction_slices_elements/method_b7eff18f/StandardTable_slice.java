// Source-based slice around line 803
// Method: <com.google.common.collect.StandardTable: Map rowMap()>

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

  Map<R, Map<C, V>> createRowMap() {
    return new RowMap();
  }

  @WeakOuter
  class RowMap extends ViewCachingAbstractMap<R, Map<C, V>> {
