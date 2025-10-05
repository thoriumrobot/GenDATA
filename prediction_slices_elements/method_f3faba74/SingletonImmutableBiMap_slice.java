// Source-based slice around line 83
// Method: <com.google.common.collect.SingletonImmutableBiMap: boolean isPartialView()>

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
    return ImmutableSet.of(immutableEntry(singleKey, singleValue));
  }

  @Override
  ImmutableSet<K> createKeySet() {
