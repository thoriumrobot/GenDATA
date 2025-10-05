// Source-based slice around line 71
// Method: <com.google.common.collect.SingletonImmutableSet: int copyIntoArray(Object[],int)>

    return ImmutableList.of(element);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  int copyIntoArray(@Nullable Object[] dst, int offset) {
    dst[offset] = element;
    return offset + 1;
  }

  @Override
  public final int hashCode() {
    return element.hashCode();
  }

  @Override
