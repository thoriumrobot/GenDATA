// Source-based slice around line 154
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getAllClasses()>

  public ImmutableSet<ResourceInfo> getResources() {
    return resources;
  }

  /**
   * Returns all classes loadable from the current class path.
   *
   * @since 16.0
   */
  public ImmutableSet<ClassInfo> getAllClasses() {
    return FluentIterable.from(resources).filter(ClassInfo.class).toSet();
  }

  /**
   * Returns all top level classes loadable from the current class path. Note that "top-level-ness"
   * is determined heuristically by class name (see {@link ClassInfo#isTopLevel}).
   */
  public ImmutableSet<ClassInfo> getTopLevelClasses() {
    return FluentIterable.from(resources)
        .filter(ClassInfo.class)
