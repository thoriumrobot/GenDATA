// Source-based slice around line 148
// Method: <com.google.common.collect.CartesianList: int size()>

      @Override
      @GwtIncompatible // serialization
      Object writeReplace() {
        return super.writeReplace();
      }
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
