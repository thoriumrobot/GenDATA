// Source-based slice around line 114
// Method: <com.google.common.io.TempFileCreator: void testMakingUserPermissionsFromScratch()>

  /**
   * Creates the permissions normally used for Windows filesystems, looking up the user afresh, even
   * if previous calls have initialized the {@code PermissionSupplier} fields.
   *
   * <p>This lets us test the effects of different values of the {@code user.name} system property
   * without needing a separate VM or classloader.
   */
  @IgnoreJRERequirement // used only when Path is available (and only from tests)
  @VisibleForTesting
  static void testMakingUserPermissionsFromScratch() throws IOException {
    // All we're testing is whether it throws.
    FileAttribute<?> unused = JavaNioCreator.userPermissions().get();
  }

  @IgnoreJRERequirement // used only when Path is available
  private static final class JavaNioCreator extends TempFileCreator {
    @Override
    File createTempDir() {
      try {
        return java.nio.file.Files.createTempDirectory(
