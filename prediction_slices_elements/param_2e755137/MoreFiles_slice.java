// Source-based slice around line 263
// Method: <com.google.common.io.MoreFiles: ImmutableList listFiles(Path)>


  /**
   * Returns an immutable list of paths to the files contained in the given directory.
   *
   * @throws NoSuchFileException if the file does not exist <i>(optional specific exception)</i>
   * @throws NotDirectoryException if the file could not be opened because it is not a directory
   *     <i>(optional specific exception)</i>
   * @throws IOException if an I/O error occurs
   */
  public static ImmutableList<Path> listFiles(Path dir) throws IOException {
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
      return ImmutableList.copyOf(stream);
    } catch (DirectoryIteratorException e) {
      throw e.getCause();
    }
  }

  /**
   * Returns a {@link Traverser} instance for the file and directory tree. The returned traverser
   * starts from a {@link Path} and will return all files and directories it encounters.
