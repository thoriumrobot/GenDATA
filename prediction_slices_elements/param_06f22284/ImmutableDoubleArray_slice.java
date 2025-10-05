// Source-based slice around line 123
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray of(double,double,double,double,double,double)>

  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray of(double e0, double e1, double e2, double e3, double e4) {
    return new ImmutableDoubleArray(new double[] {e0, e1, e2, e3, e4});
  }

  /** Returns an immutable array containing the given values, in order. */
  public static ImmutableDoubleArray of(
      double e0, double e1, double e2, double e3, double e4, double e5) {
    return new ImmutableDoubleArray(new double[] {e0, e1, e2, e3, e4, e5});
  }

  // TODO(kevinb): go up to 11?

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p>The array {@code rest} must not be longer than {@code Integer.MAX_VALUE - 1}.
   */
