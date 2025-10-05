// Source-based slice around line 809
// Method: <com.google.common.collect.ImmutableMultimap: UnmodifiableIterator valueIterator()>

    return (ImmutableCollection<V>) super.values();
  }

  @Override
  ImmutableCollection<V> createValues() {
    return new Values<>(this);
  }

  @Override
  UnmodifiableIterator<V> valueIterator() {
    return new UnmodifiableIterator<V>() {
      final Iterator<? extends ImmutableCollection<V>> valueCollectionItr = map.values().iterator();
      Iterator<V> valueItr = emptyIterator();

      @Override
      public boolean hasNext() {
        return valueItr.hasNext() || valueCollectionItr.hasNext();
      }

      @Override
