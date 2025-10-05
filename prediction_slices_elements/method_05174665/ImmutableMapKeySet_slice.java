// Source-based slice around line 54
// Method: <com.google.common.collect.ImmutableMapKeySet: Spliterator spliterator()>

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
    return map.containsKey(object);
  }

  @Override
  K get(int index) {
