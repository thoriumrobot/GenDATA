// Source-based slice around line 390
// Method: <com.google.common.primitives.Bytes: void reverse(byte[])>

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  /**
   * Reverses the elements of {@code array}. This is equivalent to {@code
   * Collections.reverse(Bytes.asList(array))}, but is likely to be more efficient.
   *
   * @since 23.1
   */
  public static void reverse(byte[] array) {
    checkNotNull(array);
    reverse(array, 0, array.length);
  }

  /**
   * Reverses the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive. This is equivalent to {@code
   * Collections.reverse(Bytes.asList(array).subList(fromIndex, toIndex))}, but is likely to be more
   * efficient.
   *
