// Source-based slice around line 108
// Method: <com.google.common.collect.ImmutableEnumMap: void forEach(BiConsumer)>

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
    return false;
  }

  // All callers of the constructor are restricted to <K extends Enum<K>>.
  @Override
