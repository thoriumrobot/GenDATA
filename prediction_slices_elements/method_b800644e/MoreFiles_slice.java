// Source-based slice around line 341
// Method: <com.google.common.io.MoreFiles: Predicate isRegularFile(LinkOption)>

    return dir.getFileAttributeView(name, BasicFileAttributeView.class, options)
        .readAttributes()
        .isDirectory();
  }

  /**
   * Returns a predicate that returns the result of {@link java.nio.file.Files#isRegularFile(Path,
   * LinkOption...)} on input paths with the given link options.
   */
  public static Predicate<Path> isRegularFile(LinkOption... options) {
    LinkOption[] optionsCopy = options.clone();
    return new Predicate<Path>() {
      @Override
      public boolean apply(Path input) {
        return Files.isRegularFile(input, optionsCopy);
      }

      @Override
      public String toString() {
        return "MoreFiles.isRegularFile(" + Arrays.toString(optionsCopy) + ")";
