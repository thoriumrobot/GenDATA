// Source-based slice around line 62
// Method: <com.google.common.io.TempFileCreator: File createTempDir()>

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

  private static TempFileCreator pickSecureCreator() {
    try {
      Class.forName("java.nio.file.Path");
      return new JavaNioCreator();
    } catch (ClassNotFoundException runningUnderAndroid) {
      // Try another way.
    }
