// Source-based slice around line 202
// Method: <com.google.common.io.FileBackedOutputStream: void write(int)>

        file = null;
        if (!deleteMe.delete()) {
          throw new IOException("Could not delete: " + deleteMe);
        }
      }
    }
  }

  @Override
  public synchronized void write(int b) throws IOException {
    update(1);
    out.write(b);
  }

  @Override
  public synchronized void write(byte[] b) throws IOException {
    write(b, 0, b.length);
  }

  @Override
