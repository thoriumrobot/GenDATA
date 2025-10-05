// Source-based slice around line 44
// Method: <com.google.common.collect.testing.OpenJdk6SetTests: Collection suppressForTreeSetNatural()>

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

  @Override
  protected Collection<Method> suppressForCheckedSet() {
    return asList(getAddNullSupportedMethod(), getAddSupportedNullPresentMethod());
  }
