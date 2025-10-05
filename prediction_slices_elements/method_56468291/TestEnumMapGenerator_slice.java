// Source-based slice around line 38
// Method: <com.google.common.collect.testing.TestEnumMapGenerator: SampleElements samples()>

 * Implementation helper for {@link TestMapGenerator} for use with enum maps.
 *
 * @author Kevin Bourrillion
 */
@GwtCompatible
@NullMarked
public abstract class TestEnumMapGenerator implements TestMapGenerator<AnEnum, String> {

  @Override
  public SampleElements<Entry<AnEnum, String>> samples() {
    return new SampleElements<>(
        mapEntry(AnEnum.A, "January"),
        mapEntry(AnEnum.B, "February"),
        mapEntry(AnEnum.C, "March"),
        mapEntry(AnEnum.D, "April"),
        mapEntry(AnEnum.E, "May"));
  }

  @Override
  public final Map<AnEnum, String> create(Object... entries) {
