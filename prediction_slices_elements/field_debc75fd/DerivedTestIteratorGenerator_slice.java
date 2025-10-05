// Source-based slice around line 30
// Method: com.google.common.collect.testing.DerivedTestIteratorGenerator.collectionGenerator


/**
 * Adapts a test iterable generator to give a TestIteratorGenerator.
 *
 * @author George van den Driessche
 */
@GwtCompatible
public final class DerivedTestIteratorGenerator<E>
    implements TestIteratorGenerator<E>, DerivedGenerator {
  private final TestSubjectGenerator<? extends Iterable<E>> collectionGenerator;

  public DerivedTestIteratorGenerator(
      TestSubjectGenerator<? extends Iterable<E>> collectionGenerator) {
    this.collectionGenerator = collectionGenerator;
  }

  @Override
  public TestSubjectGenerator<? extends Iterable<E>> getInnerGenerator() {
    return collectionGenerator;
  }
