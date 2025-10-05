/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.io;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.net.URI;
    @Positive
import java.net.URL;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.nio.file.FileStore;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.Path;
    @Positive
import java.security.SecureRandom;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@CFComment({ "nullness:", "This @EnsuresNonNullIfTrue is not true, since the list methods also", "return null in the case of an IO error (instead of throwing IOException).", "EnsuresNonNullIf(expression={\"list()\",\"list(FilenameFilter)\",\"listFiles()\",\"listFiles(FilenameFilter)\",\"listFiles(FileFilter)\"}, result=true)\"" })
    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
public class File implements Serializable, Comparable<File> {

    @Positive
    @Pure
    @Positive
    final boolean isInvalid();

    @Positive
    @Pure
    @Positive
    int getPrefixLength();

    @Positive
    public static final char separatorChar;

    @Positive
    @Interned
    @Positive
    public static final String separator;

    @Positive
    public static final char pathSeparatorChar;

    @Positive
    @Interned
    @Positive
    @Regex
    @Positive
    public static final String pathSeparator;

    @Positive
    @SideEffectFree
    @Positive
    public File(String pathname) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public File(@Nullable String parent, String child) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public File(@Nullable File parent, String child) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public File(URI uri) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public String getName();

    @Positive
    @Pure
    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public String getParent(@GuardSatisfied File this);

    @Positive
    @Pure
    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public File getParentFile(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public String getPath();

    @Positive
    @Pure
    @Positive
    public boolean isAbsolute(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public String getAbsolutePath();

    @Positive
    @SideEffectFree
    @Positive
    public File getAbsoluteFile();

    @Positive
    @SideEffectFree
    @Positive
    @ReleasesNoLocks
    @Positive
    public String getCanonicalPath() throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public File getCanonicalFile() throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public URL toURL() throws MalformedURLException;

    @Positive
    @SideEffectFree
    @Positive
    public URI toURI();

    @Positive
    @SideEffectFree
    @Positive
    public boolean canRead();

    @Positive
    @SideEffectFree
    @Positive
    public boolean canWrite();

    @Positive
    @SideEffectFree
    @Positive
    public boolean exists();

    @Positive
    @SideEffectFree
    @Positive
    public boolean isDirectory(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public boolean isFile(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public boolean isHidden(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public long lastModified();

    @Positive
    @SideEffectFree
    @Positive
    @NonNegative
    @Positive
    public long length();

    @Positive
    public boolean createNewFile() throws IOException;

    @Positive
    public boolean delete();

    @Positive
    public void deleteOnExit();

    @Positive
    @SideEffectFree
    @Positive
    public String @Nullable [] list();

    @Positive
    @SideEffectFree
    @Positive
    public String @Nullable [] list(@Nullable FilenameFilter filter);

    @Positive
    @SideEffectFree
    @Positive
    public File @Nullable [] listFiles();

    @Positive
    @SideEffectFree
    @Positive
    public File @Nullable [] listFiles(@Nullable FilenameFilter filter);

    @Positive
    @SideEffectFree
    @Positive
    public File @Nullable [] listFiles(@Nullable FileFilter filter);

    @Positive
    public boolean mkdir();

    @Positive
    public boolean mkdirs();

    @Positive
    public boolean renameTo(File dest);

    @Positive
    public boolean setLastModified(long time);

    @Positive
    public boolean setReadOnly();

    @Positive
    public boolean setWritable(boolean writable, boolean ownerOnly);

    @Positive
    public boolean setWritable(boolean writable);

    @Positive
    public boolean setReadable(boolean readable, boolean ownerOnly);

    @Positive
    public boolean setReadable(boolean readable);

    @Positive
    public boolean setExecutable(boolean executable, boolean ownerOnly);

    @Positive
    public boolean setExecutable(boolean executable);

    @Positive
    @SideEffectFree
    @Positive
    public boolean canExecute();

    @Positive
    @SideEffectFree
    @Positive
    public static File @Nullable [] listRoots();

    @Positive
    @SideEffectFree
    @Positive
    @NonNegative
    @Positive
    public long getTotalSpace();

    @Positive
    @SideEffectFree
    @Positive
    @NonNegative
    @Positive
    public long getFreeSpace();

    @Positive
    @SideEffectFree
    @Positive
    @NonNegative
    @Positive
    public long getUsableSpace();

    @Positive
    private static class TempDirectory {

    @Positive
        static File location();

    @Positive
        @SuppressWarnings("removal")
    @Positive
        static File generateFile(String prefix, String suffix, File dir) throws IOException;
    @Positive
    }

    @Positive
    public static File createTempFile(String prefix, @Nullable String suffix, @Nullable File directory) throws IOException;

    @Positive
    public static File createTempFile(String prefix, @Nullable String suffix) throws IOException;

    @Positive
    @Pure
    @Positive
    public int compareTo(@GuardSatisfied File this, @GuardSatisfied File pathname);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied File this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied File this);

    @Positive
    @SideEffectFree
    @Positive
    public Path toPath();
    @Positive
}
