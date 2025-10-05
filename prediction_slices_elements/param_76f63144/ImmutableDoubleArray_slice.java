// Source-based slice around line 153
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray copyOf(Collection)>


  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray copyOf(double[] values) {
    return values.length == 0
        ? EMPTY
        : new ImmutableDoubleArray(Arrays.copyOf(values, values.length));
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray copyOf(Collection<Double> values) {
    return values.isEmpty() ? EMPTY : new ImmutableDoubleArray(Doubles.toArray(values));
  }

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p><b>Performance note:</b> this method delegates to {@link #copyOf(Collection)} if {@code
   * values} is a {@link Collection}. Otherwise it creates a {@link #builder} and uses {@link
   * Builder#addAll(Iterable)}, with all the performance implications associated with that.
   */
