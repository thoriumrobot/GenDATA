// Source-based slice around line 666
// Method: <com.google.common.collect.ImmutableMultimap: UnmodifiableIterator entryIterator()>

    @GwtIncompatible
        Object writeReplace() {
      return super.writeReplace();
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  @Override
  UnmodifiableIterator<Entry<K, V>> entryIterator() {
    return new UnmodifiableIterator<Entry<K, V>>() {
      final Iterator<? extends Entry<K, ? extends ImmutableCollection<V>>> asMapItr =
          map.entrySet().iterator();
      @Nullable K currentKey = null;
      Iterator<V> valueItr = emptyIterator();

      @Override
      public boolean hasNext() {
        return valueItr.hasNext() || asMapItr.hasNext();
      }
