// Source-based slice around line 34
// Method: <com.google.common.collect.testing.TestIntegerSetGenerator: SampleElements samples()>

/**
 * Create integer sets for collection tests.
 *
 * @author Gregory Kick
 */
@GwtCompatible
@NullMarked
public abstract class TestIntegerSetGenerator implements TestSetGenerator<Integer> {
  @Override
  public SampleElements<Integer> samples() {
    return new Ints();
  }

  @Override
  public Set<Integer> create(Object... elements) {
    Integer[] array = new Integer[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (Integer) e;
    }
