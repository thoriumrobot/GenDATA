// Source-based slice around line 61
// Method: <com.google.common.collect.testing.TestCharacterListGenerator: List order(List)>

  protected abstract List<Character> create(Character[] elements);

  @Override
  public Character[] createArray(int length) {
    return new Character[length];
  }

  /** Returns the original element list, unchanged. */
  @Override
  public List<Character> order(List<Character> insertionOrder) {
    return insertionOrder;
  }
}
