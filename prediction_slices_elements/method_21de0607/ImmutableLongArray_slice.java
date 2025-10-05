// Source-based slice around line 152
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray copyOf(Collection)>


  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray copyOf(long[] values) {
    return values.length == 0
        ? EMPTY
        : new ImmutableLongArray(Arrays.copyOf(values, values.length));
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableLongArray copyOf(Collection<Long> values) {
    return values.isEmpty() ? EMPTY : new ImmutableLongArray(Longs.toArray(values));
  }

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p><b>Performance note:</b> this method delegates to {@link #copyOf(Collection)} if {@code
   * values} is a {@link Collection}. Otherwise it creates a {@link #builder} and uses {@link
   * Builder#addAll(Iterable)}, with all the performance implications associated with that.
   */
