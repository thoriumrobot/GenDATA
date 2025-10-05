// Source-based slice around line 153
// Method: <com.google.common.collect.CartesianList: boolean contains(Object)>

    };
  }

  @Override
  public int size() {
    return axesSizeProduct[0];
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (!(object instanceof List)) {
      return false;
    }
    List<?> list = (List<?>) object;
    if (list.size() != axes.size()) {
      return false;
    }
    int i = 0;
    for (Object o : list) {
      if (!axes.get(i).contains(o)) {
