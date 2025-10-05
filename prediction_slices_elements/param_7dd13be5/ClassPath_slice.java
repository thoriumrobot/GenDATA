// Source-based slice around line 395
// Method: <com.google.common.reflect.ClassPath: ImmutableSet locationsFrom(ClassLoader)>

      return className;
    }
  }

  /**
   * Returns all locations that {@code classloader} and parent loaders load classes and resources
   * from. Callers can {@linkplain LocationInfo#scanResources scan} individual locations selectively
   * or even in parallel.
   */
  static ImmutableSet<LocationInfo> locationsFrom(ClassLoader classloader) {
    ImmutableSet.Builder<LocationInfo> builder = ImmutableSet.builder();
    for (Map.Entry<File, ClassLoader> entry : getClassPathEntries(classloader).entrySet()) {
      builder.add(new LocationInfo(entry.getKey(), entry.getValue()));
    }
    return builder.build();
  }

  /**
   * Represents a single location (a directory or a jar file) in the class path and is responsible
   * for scanning resources from this location.
