// Source-based slice around line 433
// Method: <com.google.common.primitives.ImmutableIntArray: IntStream stream()>

      consumer.accept(array[i]);
    }
  }

  /**
   * Returns a stream over the values in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public IntStream stream() {
    return Arrays.stream(array, start, end);
  }

  /** Returns a new, mutable copy of this array's values, as a primitive {@code int[]}. */
  public int[] toArray() {
    return Arrays.copyOfRange(array, start, end);
  }

  /**
   * Returns a new immutable array containing the values in the specified range.
