// Source-based slice around line 464
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableCollection replaceValues(K,Iterable)>

   * @deprecated Unsupported operation.
   */
  @CanIgnoreReturnValue
  @Deprecated
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  // DoNotCall wants this to be final, but we want to override it to return more specific types.
  // Inheritance is closed, and all subtypes are @DoNotCall, so this is safe to suppress.
  @SuppressWarnings("DoNotCall")
  public ImmutableCollection<V> replaceValues(K key, Iterable<? extends V> values) {
    throw new UnsupportedOperationException();
  }

  /**
   * Guaranteed to throw an exception and leave the multimap unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @Deprecated
