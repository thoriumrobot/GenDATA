// Source-based slice around line 77
// Method: <com.google.common.collect.ImmutableMapValues: boolean isPartialView()>

    return CollectSpliterators.map(map.entrySet().spliterator(), Entry::getValue);
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return object != null && Iterators.contains(iterator(), object);
  }

  @Override
  boolean isPartialView() {
    return true;
  }

  @Override
  public ImmutableList<V> asList() {
    ImmutableList<Entry<K, V>> entryList = map.entrySet().asList();
    return new ImmutableAsList<V>() {
      @Override
      public V get(int index) {
        return entryList.get(index).getValue();
