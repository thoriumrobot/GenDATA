// Source-based slice around line 98
// Method: <com.google.common.collect.ImmutableEnumMap: UnmodifiableIterator entryIterator()>

      return true;
    }
    if (object instanceof ImmutableEnumMap) {
      object = ((ImmutableEnumMap<?, ?>) object).delegate;
    }
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
