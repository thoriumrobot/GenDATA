// Source-based slice around line 145
// Method: <com.google.common.primitives.ImmutableIntArray: ImmutableIntArray copyOf(int[])>

    checkArgument(
        rest.length <= Integer.MAX_VALUE - 1, "the total number of elements must fit in an int");
    int[] array = new int[rest.length + 1];
    array[0] = first;
    System.arraycopy(rest, 0, array, 1, rest.length);
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
