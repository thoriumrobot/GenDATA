// Source-based slice around line 700
// Method: <com.google.common.collect.Range: Cut upperBound()>

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
    } else {
      return this;
    }
  }
