// Source-based slice around line 804
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableCollection createValues()>

   * Returns an immutable collection of the values in this multimap. Its iterator traverses the
   * values for the first key, the values for the second key, and so on.
   */
  @Override
  public ImmutableCollection<V> values() {
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
