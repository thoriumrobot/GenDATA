// Source-based slice around line 135
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray of(long,long)>

  // TODO(kevinb): go up to 11?

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p>The array {@code rest} must not be longer than {@code Integer.MAX_VALUE - 1}.
   */
  // Use (first, rest) so that `of(someLongArray)` won't compile (they should use copyOf), which is
  // okay since we have to copy the just-created array anyway.
  public static ImmutableLongArray of(long first, long... rest) {
    checkArgument(
        rest.length <= Integer.MAX_VALUE - 1, "the total number of elements must fit in an int");
    long[] array = new long[rest.length + 1];
    array[0] = first;
    System.arraycopy(rest, 0, array, 1, rest.length);
    return new ImmutableLongArray(array);
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray copyOf(long[] values) {
