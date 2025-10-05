// Source-based slice around line 524
// Method: <com.google.common.primitives.Booleans: int countTrue(boolean)>


    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  /**
   * Returns the number of {@code values} that are {@code true}.
   *
   * @since 16.0
   */
  public static int countTrue(boolean... values) {
    int count = 0;
    for (boolean value : values) {
      if (value) {
        count++;
      }
    }
    return count;
  }

  /**
