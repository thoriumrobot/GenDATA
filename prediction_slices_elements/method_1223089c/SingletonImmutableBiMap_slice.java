// Source-based slice around line 63
// Method: <com.google.common.collect.SingletonImmutableBiMap: int size()>

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
    checkNotNull(action).accept(singleKey, singleValue);
  }

  @Override
  public boolean containsKey(@Nullable Object key) {
