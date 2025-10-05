// Source-based slice around line 185
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getTopLevelClassesRecursive(String)>

      }
    }
    return builder.build();
  }

  /**
   * Returns all top level classes whose package name is {@code packageName} or starts with {@code
   * packageName} followed by a '.'.
   */
  public ImmutableSet<ClassInfo> getTopLevelClassesRecursive(String packageName) {
    checkNotNull(packageName);
    String packagePrefix = packageName + '.';
    ImmutableSet.Builder<ClassInfo> builder = ImmutableSet.builder();
    for (ClassInfo classInfo : getTopLevelClasses()) {
      if (classInfo.getName().startsWith(packagePrefix)) {
        builder.add(classInfo);
      }
    }
    return builder.build();
  }
