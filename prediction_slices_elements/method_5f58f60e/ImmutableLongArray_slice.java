// Source-based slice around line 175
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray copyOf(LongStream)>

    }
    return builder().addAll(values).build();
  }

  /**
   * Returns an immutable array containing all the values from {@code stream}, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public static ImmutableLongArray copyOf(LongStream stream) {
    // Note this uses very different growth behavior from copyOf(Iterable) and the builder.
    long[] array = stream.toArray();
    return (array.length == 0) ? EMPTY : new ImmutableLongArray(array);
  }

  /**
   * Returns a new, empty builder for {@link ImmutableLongArray} instances, sized to hold up to
   * {@code initialCapacity} values without resizing. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> When feasible, {@code initialCapacity} should be the exact number
