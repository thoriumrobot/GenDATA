// Source-based slice around line 115
// Method: <com.google.common.collect.CartesianList: ImmutableList get(int)>

      if (elemIndex == -1) {
        return -1;
      }
      computedIndex += elemIndex * axesSizeProduct[axisIndex + 1];
    }
    return computedIndex;
  }

  @Override
  public ImmutableList<E> get(int index) {
    checkElementIndex(index, size());
    return new ImmutableList<E>() {

      @Override
      public int size() {
        return axes.size();
      }

      @Override
      public E get(int axis) {
