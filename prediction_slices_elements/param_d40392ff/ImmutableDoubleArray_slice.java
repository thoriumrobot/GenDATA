// Source-based slice around line 592
// Method: <com.google.common.primitives.ImmutableDoubleArray: boolean areEqual(double,double)>

    for (int i = 0; i < length(); i++) {
      if (!areEqual(this.get(i), that.get(i))) {
        return false;
      }
    }
    return true;
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
