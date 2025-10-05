// Source-based slice around line 71
// Method: <com.google.common.collect.CartesianList: int indexOf(Object)>

    }
    this.axesSizeProduct = axesSizeProduct;
  }

  private int getAxisIndexForProductIndex(int index, int axis) {
    return (index / axesSizeProduct[axis + 1]) % axes.get(axis).size();
  }

  @Override
  public int indexOf(@Nullable Object o) {
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
