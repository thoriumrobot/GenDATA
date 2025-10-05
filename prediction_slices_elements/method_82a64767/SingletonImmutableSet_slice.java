// Source-based slice around line 77
// Method: <com.google.common.collect.SingletonImmutableSet: int hashCode()>

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
  public String toString() {
    return '[' + element.toString() + ']';
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
