/*
 * Copyright (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jdk.nio.zipfs;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.channels.FileChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.*;
import java.nio.file.DirectoryStream.Filter;
import java.nio.file.attribute.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardCopyOption.COPY_ATTRIBUTES;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

final class ZipPath implements Path {

    @Override
    public ZipPath getRoot();

    @Override
    public ZipPath getFileName();

    @Override
    public ZipPath getParent();

    @Override
    public int getNameCount();

    @Override
    public ZipPath getName(int index);

    @Override
    public ZipPath subpath(int beginIndex, int endIndex);

    @Override
    public ZipPath toRealPath(LinkOption... options) throws IOException;

    boolean isHidden();

    @Override
    public ZipPath toAbsolutePath();

    @Override
    public URI toUri();

    @Override
    public Path relativize(Path other);

    @Override
    public ZipFileSystem getFileSystem();

    @Override
    public boolean isAbsolute();

    @Override
    public ZipPath resolve(Path other);

    @Override
    public Path resolveSibling(Path other);

    @Override
    public boolean startsWith(Path other);

    @Override
    public boolean endsWith(Path other);

    @Override
    public ZipPath resolve(String other);

    @Override
    public final Path resolveSibling(String other);

    @Override
    public final boolean startsWith(String other);

    @Override
    public final boolean endsWith(String other);

    @Override
    public Path normalize();

    byte[] getResolvedPath();

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int compareTo(Path other);

    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers);

    @Override
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>... events);

    @Override
    public final File toFile();

    @Override
    public Iterator<Path> iterator();

    @SuppressWarnings("unchecked")
    <V extends FileAttributeView> V getFileAttributeView(Class<V> type);

    void createDirectory(FileAttribute<?>... attrs) throws IOException;

    InputStream newInputStream(OpenOption... options) throws IOException;

    DirectoryStream<Path> newDirectoryStream(Filter<? super Path> filter) throws IOException;

    void delete() throws IOException;

    ZipFileAttributes readAttributes() throws IOException;

    @SuppressWarnings("unchecked")
    <A extends BasicFileAttributes> A readAttributes(Class<A> type) throws IOException;

    void setAttribute(String attribute, Object value, LinkOption... options) throws IOException;

    void setTimes(FileTime mtime, FileTime atime, FileTime ctime) throws IOException;

    void setOwner(UserPrincipal owner) throws IOException;

    void setPermissions(Set<PosixFilePermission> perms) throws IOException;

    void setGroup(GroupPrincipal group) throws IOException;

    Map<String, Object> readAttributes(String attributes, LinkOption... options) throws IOException;

    FileStore getFileStore() throws IOException;

    boolean isSameFile(Path other) throws IOException;

    SeekableByteChannel newByteChannel(Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    FileChannel newFileChannel(Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    void checkAccess(AccessMode... modes) throws IOException;

    OutputStream newOutputStream(OpenOption... options) throws IOException;

    void move(ZipPath target, CopyOption... options) throws IOException;

    void copy(ZipPath target, CopyOption... options) throws IOException;
}
