// Source-based slice around line 331
// Method: <com.google.common.io.MoreFiles: boolean isDirectory(SecureDirectoryStream,Path,LinkOption)>

      @Override
      public String toString() {
        return "MoreFiles.isDirectory(" + Arrays.toString(optionsCopy) + ")";
      }
    };
  }

  /** Returns whether or not the file with the given name in the given dir is a directory. */
  private static boolean isDirectory(
      SecureDirectoryStream<Path> dir, Path name, LinkOption... options) throws IOException {
    return dir.getFileAttributeView(name, BasicFileAttributeView.class, options)
        .readAttributes()
        .isDirectory();
  }

  /**
   * Returns a predicate that returns the result of {@link java.nio.file.Files#isRegularFile(Path,
   * LinkOption...)} on input paths with the given link options.
   */
  public static Predicate<Path> isRegularFile(LinkOption... options) {
