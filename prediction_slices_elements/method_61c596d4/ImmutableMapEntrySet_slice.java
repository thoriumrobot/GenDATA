// Source-based slice around line 96
// Method: <com.google.common.collect.ImmutableMapEntrySet: ImmutableMap map()>

    @J2ktIncompatible
    @GwtIncompatible
        Object writeReplace() {
      return super.writeReplace();
    }
  }

  ImmutableMapEntrySet() {}

  abstract ImmutableMap<K, V> map();

  @Override
  public int size() {
    return map().size();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (object instanceof Entry) {
      Entry<?, ?> entry = (Entry<?, ?>) object;
