// Source-based slice around line 78
// Method: <com.google.common.collect.SingletonImmutableBiMap: boolean containsValue(Object)>

    checkNotNull(action).accept(singleKey, singleValue);
  }

  @Override
  public boolean containsKey(@Nullable Object key) {
    return singleKey.equals(key);
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    return singleValue.equals(value);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  ImmutableSet<Entry<K, V>> createEntrySet() {
