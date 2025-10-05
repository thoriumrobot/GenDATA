// Source-based slice around line 53
// Method: <com.google.common.collect.testing.TestEnumSetGenerator: AnEnum[] createArray(int)>

    for (Object e : elements) {
      array[i++] = (AnEnum) e;
    }
    return create(array);
  }

  protected abstract Set<AnEnum> create(AnEnum[] elements);

  @Override
  public AnEnum[] createArray(int length) {
    return new AnEnum[length];
  }

  /** Sorts the enums according to their natural ordering. */
  /*
   * While the current implementation returns `this`, that's not something we mean to guarantee.
   * Callers of TestContainerGenerator.order need to be prepared for implementations to return a new
   * collection.
   */
  @SuppressWarnings("CanIgnoreReturnValueSuggester")
