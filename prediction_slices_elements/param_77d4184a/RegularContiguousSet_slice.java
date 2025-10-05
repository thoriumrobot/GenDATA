// Source-based slice around line 172
// Method: <com.google.common.collect.RegularContiguousSet: boolean contains(Object)>

  }

  @Override
  public int size() {
    long distance = domain.distance(first(), last());
    return (distance >= Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) distance + 1;
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (object == null) {
      return false;
    }
    try {
      @SuppressWarnings("unchecked") // The worst case is usually CCE, which we catch.
      C c = (C) object;
      return range.contains(c);
    } catch (ClassCastException e) {
      return false;
    }
