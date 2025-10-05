// Source-based slice around line 549
// Method: <com.google.common.collect.ImmutableMultimap: boolean remove(Object,Object)>

   * Guaranteed to throw an exception and leave the multimap unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @CanIgnoreReturnValue
  @Deprecated
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  public final boolean remove(@Nullable Object key, @Nullable Object value) {
    throw new UnsupportedOperationException();
  }

  /**
   * Returns {@code true} if this immutable multimap's implementation contains references to
   * user-created objects that aren't accessible via this multimap's methods. This is generally used
   * to determine whether {@code copyOf} implementations should make an explicit copy to avoid
   * memory leaks.
   */
  boolean isPartialView() {
