// Source-based slice around line 122
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray of(long,long,long,long,long,long)>

    return new ImmutableLongArray(new long[] {e0, e1, e2, e3});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray of(long e0, long e1, long e2, long e3, long e4) {
    return new ImmutableLongArray(new long[] {e0, e1, e2, e3, e4});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray of(long e0, long e1, long e2, long e3, long e4, long e5) {
    return new ImmutableLongArray(new long[] {e0, e1, e2, e3, e4, e5});
  }

  // TODO(kevinb): go up to 11?

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p>The array {@code rest} must not be longer than {@code Integer.MAX_VALUE - 1}.
   */
