// Source-based slice around line 435
// Method: <com.google.common.primitives.ImmutableLongArray: LongStream stream()>

      consumer.accept(array[i]);
    }
  }

  /**
   * Returns a stream over the values in this array, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public LongStream stream() {
    return Arrays.stream(array, start, end);
  }

  /** Returns a new, mutable copy of this array's values, as a primitive {@code long[]}. */
  public long[] toArray() {
    return Arrays.copyOfRange(array, start, end);
  }

  /**
   * Returns a new immutable array containing the values in the specified range.
