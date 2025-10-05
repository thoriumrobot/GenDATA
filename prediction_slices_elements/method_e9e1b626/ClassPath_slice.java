// Source-based slice around line 170
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getTopLevelClasses(String)>

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
    for (ClassInfo classInfo : getTopLevelClasses()) {
      if (classInfo.getPackageName().equals(packageName)) {
        builder.add(classInfo);
      }
    }
    return builder.build();
  }

