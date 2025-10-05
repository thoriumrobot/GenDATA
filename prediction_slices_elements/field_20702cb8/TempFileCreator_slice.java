// Source-based slice around line 55
// Method: com.google.common.io.TempFileCreator.INSTANCE

 * Creates temporary files and directories whose permissions are restricted to the current user or,
 * in the case of Android, the current app. If that is not possible (as is the case under the very
 * old Android Ice Cream Sandwich release), then this class throws an exception instead of creating
 * a file or directory that would be more accessible.
 */
@J2ktIncompatible
@GwtIncompatible
@J2ObjCIncompatible
abstract class TempFileCreator {
  static final TempFileCreator INSTANCE = pickSecureCreator();

  /**
   * @throws IllegalStateException if the directory could not be created (to implement the contract
   *     of {@link Files#createTempDir()}, such as if the system does not support creating temporary
   *     directories securely
   */
  abstract File createTempDir();

  abstract File createTempFile(String prefix) throws IOException;

