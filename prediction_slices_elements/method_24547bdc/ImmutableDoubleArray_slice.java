// Source-based slice around line 612
// Method: <com.google.common.primitives.ImmutableDoubleArray: String toString()>

    }
    return hash;
  }

  /**
   * Returns a string representation of this array in the same form as {@link
   * Arrays#toString(double[])}, for example {@code "[1, 2, 3]"}.
   */
  @Override
  public String toString() {
    if (isEmpty()) {
      return "[]";
    }
    StringBuilder builder = new StringBuilder(length() * 5); // rough estimate is fine
    builder.append('[').append(array[start]);

    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    builder.append(']');
