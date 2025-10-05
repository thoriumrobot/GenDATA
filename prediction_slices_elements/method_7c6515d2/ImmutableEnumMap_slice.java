// Source-based slice around line 103
// Method: <com.google.common.collect.ImmutableEnumMap: Spliterator entrySpliterator()>

    return delegate.equals(object);
  }

  @Override
  UnmodifiableIterator<Entry<K, V>> entryIterator() {
    return Maps.unmodifiableEntryIterator(delegate.entrySet().iterator());
  }

  @Override
  Spliterator<Entry<K, V>> entrySpliterator() {
    return CollectSpliterators.map(delegate.entrySet().spliterator(), Maps::unmodifiableEntry);
  }

  @Override
  public void forEach(BiConsumer<? super K, ? super V> action) {
    delegate.forEach(action);
  }

  @Override
  boolean isPartialView() {
