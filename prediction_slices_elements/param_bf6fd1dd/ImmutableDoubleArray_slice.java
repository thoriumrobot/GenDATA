// Source-based slice around line 146
// Method: <com.google.common.primitives.ImmutableDoubleArray: ImmutableDoubleArray copyOf(double[])>

    checkArgument(
        rest.length <= Integer.MAX_VALUE - 1, "the total number of elements must fit in an int");
    double[] array = new double[rest.length + 1];
    array[0] = first;
    System.arraycopy(rest, 0, array, 1, rest.length);
    return new ImmutableDoubleArray(array);
  }

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

