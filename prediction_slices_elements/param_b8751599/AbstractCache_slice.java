// Source-based slice around line 87
// Method: <com.google.common.cache.AbstractCache: void put(K,V)>

      }
    }
    return ImmutableMap.copyOf(result);
  }

  /**
   * @since 11.0
   */
  @Override
  public void put(K key, V value) {
    throw new UnsupportedOperationException();
  }

  /**
   * @since 12.0
   */
  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    for (Entry<? extends K, ? extends V> entry : m.entrySet()) {
      put(entry.getKey(), entry.getValue());
