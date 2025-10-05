// Source-based slice around line 93
// Method: com.google.common.reflect.ClassPath.logger

 *
 * <p>In the case of directory classloaders, symlinks are supported but cycles are not traversed.
 * This guarantees discovery of each <em>unique</em> loadable resource. However, not all possible
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
