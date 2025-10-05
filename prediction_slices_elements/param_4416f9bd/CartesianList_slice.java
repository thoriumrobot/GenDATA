// Source-based slice around line 93
// Method: <com.google.common.collect.CartesianList: int lastIndexOf(Object)>

      if (elemIndex == -1) {
        return -1;
      }
      computedIndex += elemIndex * axesSizeProduct[axisIndex + 1];
    }
    return computedIndex;
  }

  @Override
  public int lastIndexOf(@Nullable Object o) {
    if (!(o instanceof List)) {
      return -1;
    }
    List<?> list = (List<?>) o;
    if (list.size() != axes.size()) {
      return -1;
    }
    ListIterator<?> itr = list.listIterator();
    int computedIndex = 0;
    while (itr.hasNext()) {
