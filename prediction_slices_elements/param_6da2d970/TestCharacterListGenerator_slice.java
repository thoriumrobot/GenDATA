// Source-based slice around line 39
// Method: <com.google.common.collect.testing.TestCharacterListGenerator: List create(Object)>

@GwtCompatible
@NullMarked
public abstract class TestCharacterListGenerator implements TestListGenerator<Character> {
  @Override
  public SampleElements<Character> samples() {
    return new Chars();
  }

  @Override
  public List<Character> create(Object... elements) {
    Character[] array = new Character[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (Character) e;
    }
    return create(array);
  }

  /**
   * Creates a new collection containing the given elements; implement this method instead of {@link
