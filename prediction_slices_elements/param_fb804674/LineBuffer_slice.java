// Source-based slice around line 90
// Method: <com.google.common.io.LineBuffer: boolean finishLine(boolean)>

        default:
          // do nothing
      }
    }
    line.append(cbuf, start, off + len - start);
  }

  /** Called when a line is complete. */
  @CanIgnoreReturnValue
  private boolean finishLine(boolean sawNewline) throws IOException {
    String separator = sawReturn ? (sawNewline ? "\r\n" : "\r") : (sawNewline ? "\n" : "");
    handleLine(line.toString(), separator);
    line = new StringBuilder();
    sawReturn = false;
    return sawNewline;
  }

  /**
   * Subclasses must call this method after finishing character processing, in order to ensure that
   * any unterminated line in the buffer is passed to {@link #handleLine}.
