// Source-based slice around line 44
// Method: <com.google.common.collect.testing.OpenJdk6ListTests: Collection suppressForArraysAsList()>

 * @author Kevin Bourrillion
 */
@AndroidIncompatible // test-suite builders
public class OpenJdk6ListTests extends TestsForListsInJavaUtil {
  public static Test suite() {
    return new OpenJdk6ListTests().allTests();
  }

  @Override
  protected Collection<Method> suppressForArraysAsList() {
    return asList(getToArrayIsPlainObjectArrayMethod());
  }

  @Override
  protected Collection<Method> suppressForCheckedList() {
    return asList(
        CollectionAddTester.getAddNullSupportedMethod(),
        getAddSupportedNullPresentMethod(),
        ListAddAtIndexTester.getAddNullSupportedMethod(),
        getSetNullSupportedMethod());
