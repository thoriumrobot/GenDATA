// Source-based slice around line 704
// Method: <com.google.common.collect.Range: Object readResolve()>


  Cut<C> lowerBound() {
    return lowerBound;
  }

  Cut<C> upperBound() {
    return upperBound;
  }

  Object readResolve() {
    if (this.equals(ALL)) {
      return all();
    } else {
      return this;
    }
  }

  @SuppressWarnings("unchecked") // this method may throw CCE
  static int compareOrThrow(Comparable left, Comparable right) {
    return left.compareTo(right);
