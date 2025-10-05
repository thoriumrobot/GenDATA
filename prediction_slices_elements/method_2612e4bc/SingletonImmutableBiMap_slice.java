// Source-based slice around line 73
// Method: <com.google.common.collect.SingletonImmutableBiMap: boolean containsKey(Object)>

    return 1;
  }

  @Override
  public void forEach(BiConsumer<? super K, ? super V> action) {
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
