// Source-based slice around line 34
// Method: <com.google.common.collect.testing.TestStringSetGenerator: SampleElements samples()>

/**
 * Create string sets for collection tests.
 *
 * @author Kevin Bourrillion
 */
@GwtCompatible
@NullMarked
public abstract class TestStringSetGenerator implements TestSetGenerator<String> {
  @Override
  public SampleElements<String> samples() {
    return new Strings();
  }

  @Override
  public Set<String> create(Object... elements) {
    String[] array = new String[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (String) e;
    }
