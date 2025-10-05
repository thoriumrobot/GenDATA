// Source-based slice around line 145
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray copyOf(long[])>

    checkArgument(
        rest.length <= Integer.MAX_VALUE - 1, "the total number of elements must fit in an int");
    long[] array = new long[rest.length + 1];
    array[0] = first;
    System.arraycopy(rest, 0, array, 1, rest.length);
    return new ImmutableLongArray(array);
  }

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

