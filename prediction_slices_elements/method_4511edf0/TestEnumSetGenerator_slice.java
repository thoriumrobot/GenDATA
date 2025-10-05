// Source-based slice around line 41
// Method: <com.google.common.collect.testing.TestEnumSetGenerator: Set create(Object)>

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
    return create(array);
  }

  protected abstract Set<AnEnum> create(AnEnum[] elements);

