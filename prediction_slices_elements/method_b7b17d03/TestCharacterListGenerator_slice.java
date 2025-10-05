// Source-based slice around line 52
// Method: <com.google.common.collect.testing.TestCharacterListGenerator: List create(Character[])>

      array[i++] = (Character) e;
    }
    return create(array);
  }

  /**
   * Creates a new collection containing the given elements; implement this method instead of {@link
   * #create(Object...)}.
   */
  protected abstract List<Character> create(Character[] elements);

  @Override
  public Character[] createArray(int length) {
    return new Character[length];
  }

  /** Returns the original element list, unchanged. */
  @Override
  public List<Character> order(List<Character> insertionOrder) {
    return insertionOrder;
