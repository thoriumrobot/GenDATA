// Source-based slice around line 96
// Method: com.google.common.reflect.ClassPath.CLASS_PATH_ATTRIBUTE_SEPARATOR

 * aliases for resources on cyclic paths will be listed.
 *
 * @author Ben Yu
 * @since 14.0
 */
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

