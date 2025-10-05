// Source-based slice around line 35
// Method: <com.google.common.reflect.Reflection: String getPackageName(Class)>

 * @since 12.0
 */
public final class Reflection {

  /**
   * Returns the package name of {@code clazz} according to the Java Language Specification (section
   * 6.7). Unlike {@link Class#getPackage}, this method only parses the class name, without
   * attempting to define the {@link Package} and hence load files.
   */
  public static String getPackageName(Class<?> clazz) {
    return getPackageName(clazz.getName());
  }

  /**
   * Returns the package name of {@code classFullName} according to the Java Language Specification
   * (section 6.7). Unlike {@link Class#getPackage}, this method only parses the class name, without
   * attempting to define the {@link Package} and hence load files.
   */
  public static String getPackageName(String classFullName) {
    int lastDot = classFullName.lastIndexOf('.');
