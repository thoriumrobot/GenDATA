// Source-based slice around line 835
// Method: <com.google.common.reflect.TypeToken: boolean equals(Object)>

        return type.getRawType().isInterface();
      }
    }
  }

  /**
   * Returns true if {@code o} is another {@code TypeToken} that represents the same {@link Type}.
   */
  @Override
  public boolean equals(@Nullable Object o) {
    if (o instanceof TypeToken) {
      TypeToken<?> that = (TypeToken<?>) o;
      return runtimeType.equals(that.runtimeType);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return runtimeType.hashCode();
