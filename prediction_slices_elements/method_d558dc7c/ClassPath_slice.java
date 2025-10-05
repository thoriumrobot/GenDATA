// Source-based slice around line 162
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getTopLevelClasses()>

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
        .filter(ClassInfo::isTopLevel)
        .toSet();
  }

  /** Returns all top level classes whose package name is {@code packageName}. */
  public ImmutableSet<ClassInfo> getTopLevelClasses(String packageName) {
    checkNotNull(packageName);
    ImmutableSet.Builder<ClassInfo> builder = ImmutableSet.builder();
