// Source-based slice around line 39
// Method: <com.google.common.collect.testing.OpenJdk6SetTests: Test suite()>


/**
 * Tests the {@link Set} implementations of {@link java.util}, suppressing tests that trip known
 * OpenJDK 6 bugs.
 *
 * @author Kevin Bourrillion
 */
@AndroidIncompatible // test-suite builders
public class OpenJdk6SetTests extends TestsForSetsInJavaUtil {
  public static Test suite() {
    return new OpenJdk6SetTests().allTests();
  }

  @Override
  protected Collection<Method> suppressForTreeSetNatural() {
    return asList(
        getAddNullUnsupportedMethod(),
        getAddAllNullUnsupportedMethod(),
        getCreateWithNullUnsupportedMethod());
  }
