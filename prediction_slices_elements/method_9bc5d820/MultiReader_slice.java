// Source-based slice around line 87
// Method: <com.google.common.io.MultiReader: void close()>

    return 0;
  }

  @Override
  public boolean ready() throws IOException {
    return (current != null) && current.ready();
  }

  @Override
  public void close() throws IOException {
    if (current != null) {
      try {
        current.close();
      } finally {
        current = null;
      }
    }
  }
}
