// Source-based slice around line 39
// Method: <com.google.common.collect.testing.OpenJdk6ListTests: Test suite()>


/**
 * Tests the {@link List} implementations of {@link java.util}, suppressing tests that trip known
 * OpenJDK 6 bugs.
 *
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
