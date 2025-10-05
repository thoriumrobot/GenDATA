// Source-based slice around line 696
// Method: <com.google.common.collect.Range: Cut lowerBound()>

    StringBuilder sb = new StringBuilder(16);
    lowerBound.describeAsLowerBound(sb);
    sb.append("..");
    upperBound.describeAsUpperBound(sb);
    return sb.toString();
  }

  // We declare accessors so that we can use method references like `Range::lowerBound`.

  Cut<C> lowerBound() {
    return lowerBound;
  }

  Cut<C> upperBound() {
    return upperBound;
  }

  Object readResolve() {
    if (this.equals(ALL)) {
      return all();
