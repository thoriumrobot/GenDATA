// Source-based slice around line 390
// Method: <com.google.common.io.MoreFiles: void touch(Path)>

    }
    return source1.contentEquals(source2);
  }

  /**
   * Like the unix command of the same name, creates an empty file or updates the last modified
   * timestamp of the existing file at the given path to the current system time.
   */
  @SuppressWarnings("GoodTime") // reading system time without TimeSource
  public static void touch(Path path) throws IOException {
    checkNotNull(path);

    try {
      Files.setLastModifiedTime(path, FileTime.fromMillis(System.currentTimeMillis()));
    } catch (NoSuchFileException e) {
      try {
        Files.createFile(path);
      } catch (FileAlreadyExistsException ignore) {
        // The file didn't exist when we called setLastModifiedTime, but it did when we called
        // createFile, so something else created the file in between. The end result is
