// Source-based slice around line 722
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultiset keys()>

            (key, valueCollection) -> valueCollection.forEach(value -> action.accept(key, value)));
  }

  /**
   * Returns an immutable multiset containing all the keys in this multimap, in the same order and
   * with the same frequencies as they appear in this multimap; to get only a single occurrence of
   * each key, use {@link #keySet}.
   */
  @Override
  public ImmutableMultiset<K> keys() {
    return (ImmutableMultiset<K>) super.keys();
  }

  @Override
  ImmutableMultiset<K> createKeys() {
    return new Keys();
  }

  @SuppressWarnings("serial") // Uses writeReplace, not default serialization
  @WeakOuter
