// Source-based slice around line 727
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultiset createKeys()>

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
  private final class Keys extends ImmutableMultiset<K> {
    @Override
    public boolean contains(@Nullable Object object) {
      return containsKey(object);
    }
