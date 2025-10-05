// Source-based slice around line 46
// Method: <com.google.common.collect.testing.OpenJdk6MapTests: Test suite()>

/**
 * Tests the {@link Map} implementations of {@link java.util}, suppressing tests that trip known
 * bugs in OpenJDK 6 or higher.
 *
 * @author Kevin Bourrillion
 */
// TODO(cpovirk): consider renaming this class in light of our now running it under newer JDKs.
@AndroidIncompatible // test-suite builders
public class OpenJdk6MapTests extends TestsForMapsInJavaUtil {
  public static Test suite() {
    return new OpenJdk6MapTests().allTests();
  }

  @Override
  protected Collection<Method> suppressForTreeMapNatural() {
    return asList(
        getPutNullKeyUnsupportedMethod(),
        getPutAllNullKeyUnsupportedMethod(),
        getCreateWithNullKeyUnsupportedMethod(),
        getCreateWithNullUnsupportedMethod(), // for keySet
