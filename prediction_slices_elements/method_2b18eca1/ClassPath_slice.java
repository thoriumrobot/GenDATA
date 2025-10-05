// Source-based slice around line 145
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getResources()>

      builder.addAll(location.scanResources(scanned));
    }
    return new ClassPath(builder.build());
  }

  /**
   * Returns all resources loadable from the current class path, including the class files of all
   * loadable classes but excluding the "META-INF/MANIFEST.MF" file.
   */
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
