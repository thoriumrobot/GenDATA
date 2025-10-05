// Source-based slice around line 108
// Method: <com.google.common.primitives.Booleans: int hashCode(boolean)>


  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link
   * Boolean#hashCode(boolean)}.
   *
   * @param value a primitive {@code boolean} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Boolean.hashCode(value)")
  public static int hashCode(boolean value) {
    return Boolean.hashCode(value);
  }

  /**
   * Compares the two specified {@code boolean} values in the standard way ({@code false} is
   * considered less than {@code true}). The sign of the value returned is the same as that of
   * {@code ((Boolean) a).compareTo(b)}.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Boolean#compare} method instead.
