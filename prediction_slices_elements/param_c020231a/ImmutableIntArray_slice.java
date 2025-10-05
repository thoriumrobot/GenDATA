// Source-based slice around line 173
// Method: <com.google.common.primitives.ImmutableIntArray: ImmutableIntArray copyOf(IntStream)>

    }
    return builder().addAll(values).build();
  }

  /**
   * Returns an immutable array containing all the values from {@code stream}, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public static ImmutableIntArray copyOf(IntStream stream) {
    // Note this uses very different growth behavior from copyOf(Iterable) and the builder.
    int[] array = stream.toArray();
    return (array.length == 0) ? EMPTY : new ImmutableIntArray(array);
  }

  /**
   * Returns a new, empty builder for {@link ImmutableIntArray} instances, sized to hold up to
   * {@code initialCapacity} values without resizing. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> When feasible, {@code initialCapacity} should be the exact number
