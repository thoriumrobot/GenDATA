// Source-based slice around line 150
// Method: <com.google.common.primitives.ImmutableIntArray: ImmutableIntArray copyOf(Collection)>

    return new ImmutableIntArray(array);
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableIntArray copyOf(int[] values) {
    return values.length == 0 ? EMPTY : new ImmutableIntArray(Arrays.copyOf(values, values.length));
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableIntArray copyOf(Collection<Integer> values) {
    return values.isEmpty() ? EMPTY : new ImmutableIntArray(Ints.toArray(values));
  }

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p><b>Performance note:</b> this method delegates to {@link #copyOf(Collection)} if {@code
   * values} is a {@link Collection}. Otherwise it creates a {@link #builder} and uses {@link
   * Builder#addAll(Iterable)}, with all the performance implications associated with that.
   */
