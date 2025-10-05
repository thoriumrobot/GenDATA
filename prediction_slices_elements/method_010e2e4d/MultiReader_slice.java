// Source-based slice around line 82
// Method: <com.google.common.io.MultiReader: boolean ready()>

          return result;
        }
        advance();
      }
    }
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
