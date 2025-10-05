// Source-based slice around line 622
// Method: <com.google.common.reflect.ClassPath: ImmutableList getClassLoaderUrls(ClassLoader)>

        File file = toFile(url);
        if (!entries.containsKey(file)) {
          entries.put(file, classloader);
        }
      }
    }
    return ImmutableMap.copyOf(entries);
  }

  private static ImmutableList<URL> getClassLoaderUrls(ClassLoader classloader) {
    if (classloader instanceof URLClassLoader) {
      return ImmutableList.copyOf(((URLClassLoader) classloader).getURLs());
    }
    if (classloader.equals(ClassLoader.getSystemClassLoader())) {
      return parseJavaClassPath();
    }
    return ImmutableList.of();
  }

  /**
