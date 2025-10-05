// Source-based slice around line 49
// Method: <com.google.common.collect.ImmutableMapKeySet: UnmodifiableIterator iterator()>

    this.map = map;
  }

  @Override
  public int size() {
    return map.size();
  }

  @Override
  public UnmodifiableIterator<K> iterator() {
    return map.keyIterator();
  }

  @Override
  public Spliterator<K> spliterator() {
    return map.keySpliterator();
  }

  @Override
  public boolean contains(@Nullable Object object) {
