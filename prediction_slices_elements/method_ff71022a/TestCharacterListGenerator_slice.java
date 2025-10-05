// Source-based slice around line 34
// Method: <com.google.common.collect.testing.TestCharacterListGenerator: SampleElements samples()>

 * Generates {@code List<Character>} instances for test suites.
 *
 * @author Kevin Bourrillion
 * @author Louis Wasserman
 */
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
