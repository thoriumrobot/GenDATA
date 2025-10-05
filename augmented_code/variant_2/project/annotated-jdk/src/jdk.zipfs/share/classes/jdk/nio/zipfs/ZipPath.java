/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package jdk.nio.zipfs;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.net.URI;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import java.nio.channels.SeekableByteChannel;
    @Positive
import java.nio.file.*;
    @Positive
import java.nio.file.DirectoryStream.Filter;
    @Positive
import java.nio.file.attribute.*;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import static java.nio.charset.StandardCharsets.UTF_8;
    @Positive
import static java.nio.file.StandardCopyOption.COPY_ATTRIBUTES;
    @Positive
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
    @Positive
import static java.nio.file.StandardOpenOption.CREATE;
    @Positive
import static java.nio.file.StandardOpenOption.READ;
    @Positive
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
    @Positive
import static java.nio.file.StandardOpenOption.WRITE;

    @Positive
final class ZipPath implements Path {

    @Positive
    @Override
    @Positive
    public ZipPath getRoot();

    @Positive
    @Override
    @Positive
    public ZipPath getFileName();

    @Positive
    @Override
    @Positive
    public ZipPath getParent();

    @Positive
    @Override
    @Positive
    public int getNameCount();

    @Positive
    @Override
    @Positive
    public ZipPath getName(int index);

    @Positive
    @Override
    @Positive
    public ZipPath subpath(int beginIndex, int endIndex);

    @Positive
    @Override
    @Positive
    public ZipPath toRealPath(LinkOption... options) throws IOException;

    @Positive
    boolean isHidden();

    @Positive
    @Override
    @Positive
    public ZipPath toAbsolutePath();

    @Positive
    @Override
    @Positive
    public URI toUri();

    @Positive
    @Override
    @Positive
    public Path relativize(Path other);

    @Positive
    @Override
    @Positive
    public ZipFileSystem getFileSystem();

    @Positive
    @Override
    @Positive
    public boolean isAbsolute();

    @Positive
    @Override
    @Positive
    public ZipPath resolve(Path other);

    @Positive
    @Override
    @Positive
    public Path resolveSibling(Path other);

    @Positive
    @Override
    @Positive
    public boolean startsWith(Path other);

    @Positive
    @Override
    @Positive
    public boolean endsWith(Path other);

    @Positive
    @Override
    @Positive
    public ZipPath resolve(String other);

    @Positive
    @Override
    @Positive
    public final Path resolveSibling(String other);

    @Positive
    @Override
    @Positive
    public final boolean startsWith(String other);

    @Positive
    @Override
    @Positive
    public final boolean endsWith(String other);

    @Positive
    @Override
    @Positive
    public Path normalize();

    @Positive
    byte[] getResolvedPath();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int compareTo(Path other);

    @Positive
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers);

    @Positive
    @Override
    @Positive
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>... events);

    @Positive
    @Override
    @Positive
    public final File toFile();

    @Positive
    @Override
    @Positive
    public Iterator<Path> iterator();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    <V extends FileAttributeView> V getFileAttributeView(Class<V> type);

    @Positive
    void createDirectory(FileAttribute<?>... attrs) throws IOException;

    @Positive
    InputStream newInputStream(OpenOption... options) throws IOException;

    @Positive
    DirectoryStream<Path> newDirectoryStream(Filter<? super Path> filter) throws IOException;

    @Positive
    void delete() throws IOException;

    @Positive
    ZipFileAttributes readAttributes() throws IOException;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    <A extends BasicFileAttributes> A readAttributes(Class<A> type) throws IOException;

    @Positive
    void setAttribute(String attribute, Object value, LinkOption... options) throws IOException;

    @Positive
    void setTimes(FileTime mtime, FileTime atime, FileTime ctime) throws IOException;

    @Positive
    void setOwner(UserPrincipal owner) throws IOException;

    @Positive
    void setPermissions(Set<PosixFilePermission> perms) throws IOException;

    @Positive
    void setGroup(GroupPrincipal group) throws IOException;

    @Positive
    Map<String, Object> readAttributes(String attributes, LinkOption... options) throws IOException;

    @Positive
    FileStore getFileStore() throws IOException;

    @Positive
    boolean isSameFile(Path other) throws IOException;

    @Positive
    SeekableByteChannel newByteChannel(Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    FileChannel newFileChannel(Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    void checkAccess(AccessMode... modes) throws IOException;

    @Positive
    OutputStream newOutputStream(OpenOption... options) throws IOException;

    @Positive
    void move(ZipPath target, CopyOption... options) throws IOException;

    @Positive
    void copy(ZipPath target, CopyOption... options) throws IOException;
    @Positive
}
