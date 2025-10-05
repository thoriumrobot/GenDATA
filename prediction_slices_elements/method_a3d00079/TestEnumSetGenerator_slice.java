// Source-based slice around line 36
// Method: <com.google.common.collect.testing.TestEnumSetGenerator: SampleElements samples()>

/**
 * An abstract TestSetGenerator for generating sets containing enum values.
 *
 * @author Kevin Bourrillion
 */
@GwtCompatible
@NullMarked
public abstract class TestEnumSetGenerator implements TestSetGenerator<AnEnum> {
  @Override
  public SampleElements<AnEnum> samples() {
    return new Enums();
  }

  @Override
  public Set<AnEnum> create(Object... elements) {
    AnEnum[] array = new AnEnum[elements.length];
    int i = 0;
    for (Object e : elements) {
      array[i++] = (AnEnum) e;
    }
