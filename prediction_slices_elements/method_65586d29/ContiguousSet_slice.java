// Source-based slice around line 265
// Method: <com.google.common.collect.ContiguousSet: ImmutableSortedSet builder()>

   * Not supported. {@code ContiguousSet} instances are constructed with {@link #create}. This
   * method exists only to hide {@link ImmutableSet#builder} from consumers of {@code
   * ContiguousSet}.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Use {@link #create}.
   */
  @Deprecated
  @DoNotCall("Always throws UnsupportedOperationException")
  public static <E> ImmutableSortedSet.Builder<E> builder() {
    throw new UnsupportedOperationException();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @J2ktIncompatible // serialization
  @Override
  @GwtIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
