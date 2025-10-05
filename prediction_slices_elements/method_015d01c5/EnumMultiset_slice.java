// Source-based slice around line 100
// Method: <com.google.common.collect.EnumMultiset: boolean isActuallyE(Object)>


  /** Creates an empty {@code EnumMultiset}. */
  private EnumMultiset(Class<E> type) {
    this.type = type;
    checkArgument(type.isEnum());
    this.enumConstants = type.getEnumConstants();
    this.counts = new int[enumConstants.length];
  }

  private boolean isActuallyE(@Nullable Object o) {
    if (o instanceof Enum) {
      Enum<?> e = (Enum<?>) o;
      int index = e.ordinal();
      return index < enumConstants.length && enumConstants[index] == e;
    }
    return false;
  }

  /**
   * Returns {@code element} cast to {@code E}, if it actually is a nonnull E. Otherwise, throws
