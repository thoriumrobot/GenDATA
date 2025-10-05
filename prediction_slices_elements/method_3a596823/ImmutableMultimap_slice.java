// Source-based slice around line 799
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableCollection values()>

      return multimap.keys();
    }
  }

  /**
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
