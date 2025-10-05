// Source-based slice around line 102
// Method: <com.google.common.cache.AbstractCache: void cleanUp()>

   */
  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    for (Entry<? extends K, ? extends V> entry : m.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  @Override
  public void cleanUp() {}

  @Override
  public long size() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void invalidate(Object key) {
    throw new UnsupportedOperationException();
  }
