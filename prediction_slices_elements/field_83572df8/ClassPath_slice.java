// Source-based slice around line 101
// Method: com.google.common.reflect.ClassPath.resources

public final class ClassPath {
  private static final Logger logger = Logger.getLogger(ClassPath.class.getName());

  /** Separator for the Class-Path manifest attribute value in jar files. */
  private static final Splitter CLASS_PATH_ATTRIBUTE_SEPARATOR =
      Splitter.on(" ").omitEmptyStrings();

  private static final String CLASS_FILE_NAME_EXTENSION = ".class";

  private final ImmutableSet<ResourceInfo> resources;

  private ClassPath(ImmutableSet<ResourceInfo> resources) {
    this.resources = resources;
  }

  /**
   * Returns a {@code ClassPath} representing all classes and resources loadable from {@code
   * classloader} and its ancestor class loaders.
   *
   * <p><b>Warning:</b> {@code ClassPath} can find classes and resources only from:
