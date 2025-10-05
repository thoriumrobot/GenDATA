// Source-based slice around line 709
// Method: <com.google.common.io.MoreFiles: Path getParentPath(Path)>

      return addException(exceptions, e.getCause());
    }
  }

  /**
   * Returns a path to the parent directory of the given path. If the path actually has a parent
   * path, this is simple. Otherwise, we need to do some trickier things. Returns null if the path
   * is a root or is the empty path.
   */
  private static @Nullable Path getParentPath(Path path) {
    Path parent = path.getParent();

    // Paths that have a parent:
    if (parent != null) {
      // "/foo" ("/")
      // "foo/bar" ("foo")
      // "C:\foo" ("C:\")
      // "\foo" ("\" - current drive for process on Windows)
      // "C:foo" ("C:" - working dir of drive C on Windows)
      return parent;
