// Source-based slice around line 56
// Method: <com.google.common.collect.testing.UnhashableObject: int compareTo(UnhashableObject)>

  }

  // needed because otherwise Object.toString() calls hashCode()
  @Override
  public String toString() {
    return "DontHashMe" + value;
  }

  @Override
  public int compareTo(UnhashableObject o) {
    return Integer.compare(this.value, o.value);
  }
}
