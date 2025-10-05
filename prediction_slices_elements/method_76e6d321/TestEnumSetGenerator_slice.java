// Source-based slice around line 50
// Method: <com.google.common.collect.testing.TestEnumSetGenerator: Set create(AnEnum[])>

  public Set<AnEnum> create(Object... elements) {
    AnEnum[] array = new AnEnum[elements.length];
    int i = 0;
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
