// Source-based slice around line 328
// Method: <com.google.common.primitives.Floats: Converter stringConverter()>

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 1;
  }

  /**
   * Returns a serializable converter object that converts between strings and floats using {@link
   * Float#valueOf} and {@link Float#toString()}.
   *
   * @since 16.0
   */
  public static Converter<String, Float> stringConverter() {
    return FloatConverter.INSTANCE;
  }

  /**
   * Returns an array containing the same values as {@code array}, but guaranteed to be of a
   * specified minimum length. If {@code array} already has a length of at least {@code minLength},
   * it is returned directly. Otherwise, a new array of size {@code minLength + padding} is
   * returned, containing the values of {@code array}, and zeroes in the remaining places.
   *
   * @param array the source array
