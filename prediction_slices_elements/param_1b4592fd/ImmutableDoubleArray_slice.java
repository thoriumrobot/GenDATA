// Source-based slice around line 136
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray of(double,double)>

  // TODO(kevinb): go up to 11?

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p>The array {@code rest} must not be longer than {@code Integer.MAX_VALUE - 1}.
   */
  // Use (first, rest) so that `of(someDoubleArray)` won't compile (they should use copyOf), which
  // is okay since we have to copy the just-created array anyway.
  public static ImmutableDoubleArray of(double first, double... rest) {
    checkArgument(
        rest.length <= Integer.MAX_VALUE - 1, "the total number of elements must fit in an int");
    double[] array = new double[rest.length + 1];
    array[0] = first;
    System.arraycopy(rest, 0, array, 1, rest.length);
    return new ImmutableDoubleArray(array);
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray copyOf(double[] values) {
