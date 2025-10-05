// Source-based slice around line 598
// Method: <com.google.common.primitives.ImmutableDoubleArray: int hashCode()>

  }

  // Match the behavior of Double.equals()
  private static boolean areEqual(double a, double b) {
    return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
  }

  /** Returns an unspecified hash code for the contents of this immutable array. */
  @Override
  public int hashCode() {
    int hash = 1;
    for (int i = start; i < end; i++) {
      hash *= 31;
      hash += Double.hashCode(array[i]);
    }
    return hash;
  }

  /**
   * Returns a string representation of this array in the same form as {@link
