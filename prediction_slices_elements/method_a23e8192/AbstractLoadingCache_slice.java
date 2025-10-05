// Source-based slice around line 75
// Method: <com.google.common.cache.AbstractLoadingCache: void refresh(K)>

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
