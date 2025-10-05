// Source-based slice around line 105
// Method: <com.google.common.cache.AbstractCache: long size()>

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

  /**
   * @since 11.0
