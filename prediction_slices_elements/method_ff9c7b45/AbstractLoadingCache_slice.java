// Source-based slice around line 70
// Method: <com.google.common.cache.AbstractLoadingCache: V apply(K)>

    for (K key : keys) {
      if (!result.containsKey(key)) {
        result.put(key, get(key));
      }
    }
    return ImmutableMap.copyOf(result);
  }

  @Override
  public final V apply(K key) {
    return getUnchecked(key);
  }

  @Override
  public void refresh(K key) {
    throw new UnsupportedOperationException();
  }
}
