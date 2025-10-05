// Source-based slice around line 163
// Method: <com.google.common.primitives.ImmutableLongArray: ImmutableLongArray copyOf(Iterable)>

  }

  /**
   * Returns an immutable array containing the given values, in order.
   *
   * <p><b>Performance note:</b> this method delegates to {@link #copyOf(Collection)} if {@code
   * values} is a {@link Collection}. Otherwise it creates a {@link #builder} and uses {@link
   * Builder#addAll(Iterable)}, with all the performance implications associated with that.
   */
  public static ImmutableLongArray copyOf(Iterable<Long> values) {
    if (values instanceof Collection) {
      return copyOf((Collection<Long>) values);
    }
    return builder().addAll(values).build();
  }

  /**
   * Returns an immutable array containing all the values from {@code stream}, in order.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
