// Source-based slice around line 58
// Method: <com.google.common.collect.SingletonImmutableBiMap: V get(Object)>

  }

  private SingletonImmutableBiMap(K singleKey, V singleValue, ImmutableBiMap<V, K> inverse) {
    this.singleKey = singleKey;
    this.singleValue = singleValue;
    this.inverse = inverse;
  }

  @Override
  public @Nullable V get(@Nullable Object key) {
    return singleKey.equals(key) ? singleValue : null;
  }

  @Override
  public int size() {
    return 1;
  }

  @Override
  public void forEach(BiConsumer<? super K, ? super V> action) {
